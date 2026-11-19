/**
 * PREVIEW VAULT BINDINGS — the renderer half of a vault fill into the in-app
 * browser (`tools/browser_vault_preview.py`).
 *
 * The backend resolves the secret and builds the fill script; this window only
 * carries that script into ONE pinned guest document. A binding is an opaque
 * random target naming (tab, document generation, runner): every evaluate
 * re-checks that the tab still shows that same document, so a navigation, a
 * reload or a closed tab between the origin check and the write refuses
 * instead of filling something else. Bindings are memory-only and short-lived;
 * the script is request data and is never stored here, logged, or put in a
 * store.
 *
 * Two kinds:
 * - agent (`browser_vault` target=preview): pinned on `open` to the active tab
 *   of the requesting session; evaluated only while that tab is still the
 *   session's active tab and the session is still the one on screen.
 * - user (the Browser bar's key button): minted for the tab whose bar was
 *   clicked, before the `vault.preview.fill` RPC; the backend's evaluates come
 *   back session-less on the same socket and are matched by target alone.
 */

import { randomUuid } from '@hermes/shared'

import type { PreviewOwner } from '@/store/preview-ownership'

import {
  type ActivePreviewScriptRunner,
  isSamePreviewDocument,
  previewScriptRunnerFor,
  resolveActivePreviewScriptRunner
} from './preview-script-runner'

export interface PreviewVaultScope {
  connectionId: null | string
  profile: string
  sessionId: string
}

export interface PreviewVaultBindingInput extends PreviewVaultScope {
  target: string
}

export type PreviewVaultOpenResult = { error: string; success: false } | { success: true; target: string }

interface Binding extends PreviewVaultScope {
  document: ActivePreviewScriptRunner
  expiresAt: number
  /** Agent bindings: whose active tab was pinned (null = the user's own tab). */
  owner: null | PreviewOwner | undefined
}

const MAX_BINDINGS = 16
const AGENT_TTL_MS = 5 * 60_000
const USER_TTL_MS = 60_000
const NO_PAGE = 'No live preview page is available.'
const CHANGED = 'The selected preview page changed. Open a new target.'
const INTERRUPTED = 'The preview vault action was interrupted.'
const PAGE_FAILED = 'The preview page could not complete the vault action.'
const bindings = new Map<string, Binding>()

function prune(now = Date.now(), makeRoom = false): void {
  for (const [target, binding] of bindings) {
    if (binding.expiresAt <= now) {
      bindings.delete(target)
    }
  }

  while (makeRoom && bindings.size >= MAX_BINDINGS) {
    const oldest = bindings.keys().next().value as string | undefined

    if (!oldest) {
      return
    }

    bindings.delete(oldest)
  }
}

function pin(binding: Omit<Binding, 'expiresAt'>, ttlMs: number): string {
  prune(Date.now(), true)
  const target = randomUuid()
  bindings.set(target, { ...binding, expiresAt: Date.now() + ttlMs })

  return target
}

const isUserBinding = (binding: Binding) => binding.owner === null

/** A binding answers only the request scope it was minted for. A user binding
 *  is session-less: its evaluates come back without a session id, and the
 *  random target (never shown to the page or the model) is the capability. */
function owns(binding: Binding, input: PreviewVaultScope): boolean {
  if (isUserBinding(binding)) {
    return !input.sessionId
  }

  return (
    binding.connectionId === input.connectionId &&
    binding.profile === input.profile &&
    binding.sessionId === input.sessionId
  )
}

/** The pinned document is still live, and (agent) still the session's selected tab. */
function stillPinned(binding: Binding): boolean {
  if (!isSamePreviewDocument(binding.document)) {
    return false
  }

  return isUserBinding(binding) || resolveActivePreviewScriptRunner(binding.owner ?? undefined)?.tabId === binding.document.tabId
}

/** Agent path: pin the requesting session's active tab and its live document. */
export function openPreviewVaultBinding(scope: PreviewVaultScope, owner?: PreviewOwner): PreviewVaultOpenResult {
  const document = resolveActivePreviewScriptRunner(owner)

  return document
    ? { success: true, target: pin({ ...scope, document, owner }, AGENT_TTL_MS) }
    : { error: NO_PAGE, success: false }
}

/** User path: pin the document of the tab whose Browser bar was clicked. */
export function openUserPreviewVaultBinding(tabId: string): null | string {
  const document = previewScriptRunnerFor(tabId)

  return document ? pin({ connectionId: null, document, owner: null, profile: '', sessionId: '' }, USER_TTL_MS) : null
}

/** Run backend-generated code only while this exact document (and, for the agent, its session) still owns the binding. */
export async function evaluatePreviewVaultBinding({
  connectionId,
  expression,
  isSessionActive,
  profile,
  sessionId,
  signal,
  target
}: PreviewVaultScope & {
  expression: string
  isSessionActive: () => boolean
  signal?: AbortSignal
  target: string
}): Promise<{ decline: true } | { error: string; success: false } | { result: unknown; success: true }> {
  prune()
  const binding = bindings.get(target)

  // An unknown target may belong to a different Electron window. Let the
  // backend ask its remaining windows instead of winning with an empty error.
  if (!binding || !owns(binding, { connectionId, profile, sessionId })) {
    return { decline: true }
  }

  const stillOwned = () => stillPinned(binding) && (isUserBinding(binding) || isSessionActive())

  if (signal?.aborted) {
    return { error: INTERRUPTED, success: false }
  }

  if (!stillOwned()) {
    return { error: CHANGED, success: false }
  }

  try {
    const result = await binding.document.runner(expression, signal)

    if (signal?.aborted) {
      return { error: INTERRUPTED, success: false }
    }

    if (!stillOwned()) {
      return { error: CHANGED, success: false }
    }

    return { result: result === undefined ? null : result, success: true }
  } catch {
    // Runner errors can contain the injected script (and credential values).
    return { error: PAGE_FAILED, success: false }
  }
}

/** Release a binding after the backend's vault operation has completed. */
export function closePreviewVaultBinding(input: PreviewVaultBindingInput): { decline: true } | { success: true } {
  const binding = bindings.get(input.target)

  if (!binding || !owns(binding, input)) {
    return { decline: true }
  }

  bindings.delete(input.target)

  return { success: true }
}

/** Drop a user binding whose RPC failed before the backend could close it. */
export function forgetPreviewVaultBinding(target: string): void {
  bindings.delete(target)
}

/** Test-only state cleanup. */
export function resetPreviewVaultBindingsForTests(): void {
  bindings.clear()
}
