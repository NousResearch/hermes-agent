import {
  isActivePreviewScriptRunner,
  type PreviewScriptRunner,
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
  expiresAt: number
  generation: number
  runner: PreviewScriptRunner
  tabId: string
  target: string
}

const MAX_BINDINGS = 16
const BINDING_TTL_MS = 5 * 60_000
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

function matchesScope(binding: Binding, input: PreviewVaultScope): boolean {
  return (
    binding.connectionId === input.connectionId &&
    binding.profile === input.profile &&
    binding.sessionId === input.sessionId
  )
}

/** Pin the exact reader-selected page and runner for one vault operation. */
export function openPreviewVaultBinding(scope: PreviewVaultScope): PreviewVaultOpenResult {
  const active = resolveActivePreviewScriptRunner()

  if (!active) {
    return { error: NO_PAGE, success: false }
  }

  prune(Date.now(), true)
  const target = crypto.randomUUID()

  bindings.set(target, {
    ...scope,
    expiresAt: Date.now() + BINDING_TTL_MS,
    generation: active.generation,
    runner: active.runner,
    tabId: active.tabId,
    target
  })

  return { success: true, target }
}

/** Run backend-generated code only while this exact page and session still own the binding. */
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
  if (!binding || !matchesScope(binding, { connectionId, profile, sessionId })) {
    return { decline: true }
  }

  if (signal?.aborted) {
    return { error: INTERRUPTED, success: false }
  }

  if (!isSessionActive()) {
    return { error: CHANGED, success: false }
  }

  if (!isActivePreviewScriptRunner(binding)) {
    return { error: CHANGED, success: false }
  }

  try {
    const result = await binding.runner(expression)

    if (signal?.aborted) {
      return { error: INTERRUPTED, success: false }
    }

    if (!isSessionActive()) {
      return { error: CHANGED, success: false }
    }

    if (!isActivePreviewScriptRunner(binding)) {
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

  if (!binding || !matchesScope(binding, input)) {
    return { decline: true }
  }

  bindings.delete(input.target)

  return { success: true }
}

/** Test-only state cleanup. */
export function resetPreviewVaultBindingsForTests(): void {
  bindings.clear()
}
