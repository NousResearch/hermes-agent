import { atom, computed, type ReadableAtom } from 'nanostores'

import { $clarifyRequest, $clarifyRequests } from './clarify'
import { isSessionGone, isSessionGoneForBackgroundPolling, markSessionGone } from './runtime-gone'
import { respondToServerRequest } from './server-requests'
import { $activeSessionId } from './session'
import { setSessionsWaitingOnCard } from './session-card-wait'
import { ambientRequestFor } from './session-gone-latch'
import { requestForOwnedSession } from './session-states'

// Blocking interactive prompts the gateway raises mid-turn. Each is a
// server→client JSON-RPC request (`tui_gateway/server_requests.py`) the Python
// side blocks on; `requestId` is that request's id and the card answers through
// `store/server-requests.ts::respondToServerRequest`. Without a renderer for
// these the channel answers -32601 and the tool fails fast.
//
// Like clarify, every prompt is parked under the runtime session id that raised
// it (not one shared slot), so a *background* session running concurrently can
// raise an approval/sudo/secret prompt and have it wait — surfaced via the
// sidebar "needs input" badge — until the user switches to that chat. The
// exported $*Request view is scoped to the active session, so a background
// prompt never hijacks the foreground.

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

interface KeyedPrompt {
  sessionId: string | null
}

interface PromptStore<T extends KeyedPrompt> {
  $active: ReadableAtom<null | T>
  $all: ReadableAtom<Record<string, T>>
  clear: (sessionId?: string | null, requestId?: string) => void
  reset: () => void
  set: (request: T) => void
}

// One per-session prompt kind: a map keyed by session, plus an active-session
// view for the overlays. `clear` drops one session's entry (a request-id
// mismatch is a no-op so a stale resolve can't wipe a newer prompt); with no
// session hint it drops every entry, optionally filtered by request id.
function keyedPromptStore<T extends KeyedPrompt>(): PromptStore<T> {
  const $all = atom<Record<string, T>>({})
  const idOf = (value: T): string | undefined => (value as { requestId?: string }).requestId

  return {
    // An app-level prompt (sessionId null: the Bot Screen install card) is not about any chat, so it is
    // shown in whichever chat is active rather than only while the chat that happened to be open at
    // request time stays open.
    $active: computed([$all, $activeSessionId], (all, activeId) => all[keyFor(activeId)] ?? all[keyFor(null)] ?? null),
    $all,
    reset: () => $all.set({}),
    set: request => $all.set({ ...$all.get(), [keyFor(request.sessionId)]: request }),
    clear(sessionId, requestId) {
      const all = $all.get()

      if (sessionId !== undefined) {
        const key = keyFor(sessionId)
        const current = all[key]

        if (current && !(requestId && idOf(current) !== requestId)) {
          const next = { ...all }
          delete next[key]
          $all.set(next)
        }

        return
      }

      const next = Object.fromEntries(Object.entries(all).filter(([, v]) => requestId && idOf(v) !== requestId))

      if (Object.keys(next).length !== Object.keys(all).length) {
        $all.set(next as Record<string, T>)
      }
    }
  }
}

// Approval is queue-backed on the backend (`tools/approval.py`): `requestId` is
// the QUEUE entry's id (what `approval.pending` / `approval.received` /
// `approval.respond` key on), stable across delivery paths. The live prompt
// arrives as an `approval` server request whose id is `serverRequestId`; the
// card answers that request when it is still open and falls back to the
// `approval.respond` RPC when the prompt was restored from `approval.pending`.
export interface ApprovalRequest extends KeyedPrompt {
  // false when the backend won't honor a permanent allow → hide "Always allow".
  allowPermanent?: boolean
  choices?: string[]
  command: string
  description: string
  /** What the card asks for; absent means a command (`ApprovalRequestParams.kind`). */
  kind?: ApprovalKind
  requestId?: string
  /** One short line under the folder or site. */
  risk?: string
  serverRequestId?: string
  smartDenied?: boolean
  /** The folder (`access`) or site (`network`) the card names. */
  target?: ApprovalTarget
}

// A sandbox asks for a folder (`access`) or a site (`network`) through the same
// `approval` request and answer as a command. These cards can outlive the turn
// that raised them: the sandbox, not the agent loop, waits on the answer.
export type ApprovalKind = 'command' | 'access' | 'network'
export type ApprovalAccessLevel = 'read' | 'read_write'

export interface ApprovalAccessTarget {
  access?: ApprovalAccessLevel
  path: string
}

export interface ApprovalNetworkTarget {
  host: string
  /** A display name for the site, e.g. the Hermes tool that uses it. */
  label?: string
}

export type ApprovalTarget = ApprovalAccessTarget | ApprovalNetworkTarget

type ApprovalKindFields = Pick<ApprovalRequest, 'kind' | 'risk' | 'target'>

// Code points that draw nothing, end the line or reorder the text around them: controls, format
// characters (zero-width, bidirectional marks, overrides and isolates), unassigned and private-use
// code points, line and paragraph separators, and the blank glyphs. A card shows them as escapes,
// as the gateway does for a command (agent/redact_approval_display.py).
const HIDDEN_CHARS = /[\p{C}\p{Zl}\p{Zp}\p{Default_Ignorable_Code_Point}\u2800]/gu

function escapeHiddenChar(char: string): string {
  const code = char.codePointAt(0) ?? 0
  const hex = code.toString(16)

  return code < 0x100
    ? `\\x${hex.padStart(2, '0')}`
    : code < 0x10000
      ? `\\u${hex.padStart(4, '0')}`
      : `\\U${hex.padStart(8, '0')}`
}

/** The sender's sentence about a request, as the user reads it: trimmed, with hidden characters
 *  made visible. */
const wireText = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value.trim().replace(HIDDEN_CHARS, escapeHiddenChar) : undefined

// A space separator other than U+0020 is drawn like a space, and U+2423 is the mark `nameText`
// puts where a space would not be seen: in a name both are shown as escapes.
const LOOKALIKE_SPACES = /(?! )[\p{Zs}\u2423]/gu

/** A folder, a site or a site's label as the user reads it: the name whole, so that two names
 *  that differ only in spacing never read the same. Nothing is trimmed; a space at either end is
 *  shown as "␣". A name of white space only is no name. */
const nameText = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim()
    ? value
        .replace(HIDDEN_CHARS, escapeHiddenChar)
        .replace(LOOKALIKE_SPACES, escapeHiddenChar)
        .replace(/^ +| +$/g, spaces => '\u2423'.repeat(spaces.length))
    : undefined

/** `name` for a place that may draw a run of spaces as one blank (a system notification, a
 *  toast): each space after the first of a run becomes a no-break space, which is always drawn. */
export const withSpaceRuns = (name: string): string =>
  name.replace(/ {2,}/g, spaces => ` ${'\u00a0'.repeat(spaces.length - 1)}`)

/** The kind, target and risk an approval payload carries (live request params,
 *  `approval.pending` entries, resume snapshots). Missing or unknown kinds read
 *  as a command, so an older or newer backend never gets a blank card. */
export function approvalKindFields(raw: Record<string, unknown> | null | undefined): ApprovalKindFields {
  if (!raw || (raw.kind !== 'access' && raw.kind !== 'network')) {
    return {}
  }

  const risk = wireText(raw.risk)
  const fields: ApprovalKindFields = { kind: raw.kind, ...(risk ? { risk } : {}) }

  if (raw.kind === 'access') {
    const path = nameText(raw.path)
    const access = raw.access === 'read' || raw.access === 'read_write' ? raw.access : undefined

    return path ? { ...fields, target: { path, ...(access ? { access } : {}) } } : fields
  }

  const host = nameText(raw.host)
  const label = nameText(raw.label)

  return host ? { ...fields, target: { host, ...(label ? { label } : {}) } } : fields
}

export const isCommandApproval = (request: Pick<ApprovalRequest, 'kind'>): boolean =>
  !request.kind || request.kind === 'command'

/** A folder or site request that names no folder or site. It is refused and never parked:
 *  its card would offer Allow for something the user cannot read. */
export const approvalNamesNothing = (request: Pick<ApprovalRequest, 'kind' | 'target'>): boolean =>
  !isCommandApproval(request) && !request.target

export const accessTarget = (request: ApprovalRequest): ApprovalAccessTarget | null =>
  request.kind === 'access' && request.target && 'path' in request.target && !('host' in request.target)
    ? (request.target as ApprovalAccessTarget)
    : null

export const networkTarget = (request: ApprovalRequest): ApprovalNetworkTarget | null =>
  request.kind === 'network' && request.target && 'host' in request.target
    ? (request.target as ApprovalNetworkTarget)
    : null

/** How a network card names its site: the sender's label, else the host. */
export const networkTargetName = (target: ApprovalNetworkTarget): string => target.label || target.host

interface ApprovalGateway {
  request: (method: string, params: Record<string, unknown>) => Promise<unknown>
}

interface PendingApprovalPayload {
  access?: unknown
  allow_permanent?: boolean
  choices?: unknown
  command?: unknown
  description?: unknown
  host?: unknown
  kind?: unknown
  label?: unknown
  path?: unknown
  request_id?: unknown
  risk?: unknown
  smart_denied?: boolean
}

export interface SudoRequest extends KeyedPrompt {
  command?: string
  requestId: string
  /** Description override so the card can say WHAT the password is for (the Bot Screen install).
   *  The reply travels as a JSON-RPC response on the socket the request arrived on, so a password
   *  typed for host A can never reach host B without any origin bookkeeping here. */
  description?: string
}

export interface SecretRequest extends KeyedPrompt {
  envVar: string
  prompt: string
  requestId: string
}

// External password-manager unlock (agent/vault_backends): `vault.unlock_prompt`
// server request, answered `{value: password}`; "" keeps the manager locked.
export interface VaultUnlockRequest extends KeyedPrompt {
  backend: string
  displayName: string
  requestId: string
}

const EMPTY_APPROVALS: ApprovalRequest[] = []
const $approvalQueues = atom<Record<string, ApprovalRequest[]>>({})

// A folder or site card marks its session as waiting for as long as it is parked. Whatever
// parks or removes a card, the set of waiting sessions follows the queues.
$approvalQueues.listen(queues => {
  setSessionsWaitingOnCard(
    Object.entries(queues)
      .filter(([sessionId, queue]) => sessionId && queue.some(request => !isCommandApproval(request)))
      .map(([sessionId]) => sessionId)
  )
})
const $approvalStackSizes = atom<Record<string, number>>({})
// A replay started before a response/reset cannot resurrect the answered card.
let approvalRevision = 0
const sessionApprovalRevisions = new Map<string, number>()

const approval = {
  $all: computed($approvalQueues, queues =>
    Object.fromEntries(Object.entries(queues).map(([key, queue]) => [key, queue[0]]))
  ),
  reset() {
    approvalRevision += 1
    sessionApprovalRevisions.clear()
    $approvalStackSizes.set({})
    $approvalQueues.set({})
  },
  set(request: ApprovalRequest) {
    if (approvalNamesNothing(request)) {
      return
    }

    const key = keyFor(request.sessionId)
    const queues = $approvalQueues.get()
    const queue = queues[key] ?? EMPTY_APPROVALS
    const index = queue.findIndex(item => item.requestId === request.requestId)
    const next = [...queue]

    if (index < 0) {
      const sizes = $approvalStackSizes.get()
      $approvalStackSizes.set({ ...sizes, [key]: (sizes[key] ?? 0) + 1 })
      next.push(request)
    } else {
      next[index] = request
    }

    $approvalQueues.set({ ...queues, [key]: next })
  },
  // Drops `requestId` (or every approval) from the session's queue; `drop`
  // picks the entries instead when given.
  clear(sessionId?: string | null, requestId?: string, drop?: (item: ApprovalRequest) => boolean) {
    if (sessionId === undefined) {
      approvalRevision += 1
    } else {
      const key = keyFor(sessionId)
      sessionApprovalRevisions.set(key, (sessionApprovalRevisions.get(key) ?? 0) + 1)
    }

    const queues = $approvalQueues.get()
    const next = { ...queues }
    let changed = false

    for (const [key, queue] of Object.entries(queues)) {
      if (sessionId !== undefined && key !== keyFor(sessionId)) {
        continue
      }

      const remaining = drop
        ? queue.filter(item => !drop(item))
        : requestId
          ? queue.filter(item => item.requestId !== requestId)
          : EMPTY_APPROVALS

      if (remaining.length === queue.length) {
        continue
      }

      changed = true

      if (remaining.length) {
        next[key] = remaining
      } else {
        delete next[key]
        const sizes = { ...$approvalStackSizes.get() }
        delete sizes[key]
        $approvalStackSizes.set(sizes)
      }
    }

    if (changed) {
      $approvalQueues.set(next)
    }
  }
}

const sudo = keyedPromptStore<SudoRequest>()
const secret = keyedPromptStore<SecretRequest>()
const vaultUnlock = keyedPromptStore<VaultUnlockRequest>()

// "Save this login" for the page the agent is on (tools/browser_vault_tool): `vault.save_login`
// server request, answered `{value: JSON {identifier, password}}`; "" declines.
export interface VaultSaveLoginRequest extends KeyedPrompt {
  origin: string
  site: string
  requestId: string
}

const vaultSave = keyedPromptStore<VaultSaveLoginRequest>()

// Second-factor code for the page the agent is on: `vault.code` server request,
// answered `{value: code}`; "" skips.
export interface VaultCodeRequest extends KeyedPrompt {
  site: string
  hint: string
  requestId: string
}

const vaultCode = keyedPromptStore<VaultCodeRequest>()

export const $approvalRequests = approval.$all
export const $approvalRequest = computed(
  [approval.$all, $activeSessionId],
  (all, activeId) => all[keyFor(activeId)] ?? null
)
export const setApprovalRequest = approval.set
export const clearApprovalRequest = approval.clear

export async function receiveApprovalRequest(gateway: ApprovalGateway | null, request: ApprovalRequest): Promise<void> {
  // Nothing is parked for it, so nothing is acknowledged either.
  if (approvalNamesNothing(request)) {
    return
  }

  // A prompt restored from `approval.pending` must not clobber the live server
  // request that already carries the same queue entry (it knows how to answer).
  const current = $approvalQueues.get()[keyFor(request.sessionId)]?.find(item => item.requestId === request.requestId)

  if (
    current?.requestId &&
    current.requestId === request.requestId &&
    current.serverRequestId &&
    !request.serverRequestId
  ) {
    return
  }

  setApprovalRequest(request)

  if (gateway && request.requestId && request.sessionId) {
    try {
      await requestForOwnedSession(request.sessionId, ambientRequestFor(gateway), 'approval.received', {
        request_id: request.requestId,
        session_id: request.sessionId
      })
    } catch (error) {
      if (isSessionGoneForBackgroundPolling(error)) {
        markSessionGone(request.sessionId)

        return
      }

      throw error
    }
  }
}

export async function replayPendingApproval(gateway: ApprovalGateway | null, sessionId: string | null): Promise<void> {
  if (!gateway || !sessionId || isSessionGone(sessionId)) {
    return
  }

  const revision = approvalRevision
  const sessionRevision = sessionApprovalRevisions.get(keyFor(sessionId))
  const previous = $approvalQueues.get()[keyFor(sessionId)]
  let rawResult: unknown

  try {
    rawResult = await requestForOwnedSession(sessionId, ambientRequestFor(gateway), 'approval.pending', {
      session_id: sessionId
    })
  } catch (error) {
    if (isSessionGoneForBackgroundPolling(error)) {
      markSessionGone(sessionId)

      return
    }

    throw error
  }

  const result =
    rawResult && typeof rawResult === 'object' ? (rawResult as { approvals?: PendingApprovalPayload[] }) : {}

  if (
    revision !== approvalRevision ||
    sessionRevision !== sessionApprovalRevisions.get(keyFor(sessionId)) ||
    previous !== $approvalQueues.get()[keyFor(sessionId)]
  ) {
    return
  }

  if (!Array.isArray(result.approvals)) {
    return
  }

  const ids = new Set(result.approvals.map(pending => pending.request_id))

  for (const request of previous ?? EMPTY_APPROVALS) {
    if (request.requestId && !ids.has(request.requestId)) {
      clearApprovalRequest(sessionId, request.requestId)
    }
  }

  await Promise.all(
    result.approvals.map(pending => {
      if (typeof pending.request_id !== 'string') {
        return
      }

      return receiveApprovalRequest(gateway, {
        allowPermanent: pending.allow_permanent !== false,
        choices: Array.isArray(pending.choices)
          ? pending.choices.filter(choice => typeof choice === 'string')
          : undefined,
        command: typeof pending.command === 'string' ? pending.command : '',
        description: typeof pending.description === 'string' ? pending.description : 'dangerous command',
        requestId: pending.request_id,
        sessionId,
        smartDenied: pending.smart_denied === true,
        ...approvalKindFields(pending as Record<string, unknown>)
      })
    })
  )
}

/**
 * Resolve an approval: answer the live server request when it is still open
 * (the response frame goes back over the socket it arrived on — the owner by
 * construction), else the queue-level `approval.respond` RPC for a prompt that
 * was restored from `approval.pending` or is being answered from another
 * surface. Returns after the backend has the decision.
 */

// #55433: the backend honors an answer for the whole `approvals.timeout` window
// (default 300s), but `approval.respond` otherwise rides the generic 30s RPC
// deadline. During a long LLM stream the gateway's WS writes can stall well past
// 30s while the turn is still live and the approval is still pending — the
// client gives up with "request timed out: approval.respond" long before the
// backend would. Give the RPC a deadline that covers the backend window (300s
// plus margin for the write to drain), and retry once on a deadline failure:
// `resolve_gateway_approval` pops the queue entry before committing, so a
// duplicate resolve is a harmless `resolved: 0`.
export const APPROVAL_RESPOND_REQUEST_TIMEOUT_MS = 330_000

function isRequestTimeoutError(error: unknown): boolean {
  return error instanceof Error && /request timed out/i.test(error.message)
}

/** `reason`: why the user denied a folder or site card, passed back with the
 *  answer. Command approvals never send one. */
export async function answerApproval(
  gateway: ApprovalGateway | null,
  request: Pick<ApprovalRequest, 'requestId' | 'serverRequestId' | 'sessionId'>,
  choice: string,
  all = false,
  reason?: string
): Promise<void> {
  const why = reason?.trim() ? { reason: reason.trim() } : {}

  if (respondToServerRequest(request.serverRequestId, { choice, ...(all ? { all: true } : {}), ...why })) {
    return
  }

  if (!gateway) {
    throw new Error('Hermes gateway is not connected')
  }

  const params = {
    all,
    choice,
    ...why,
    ...(request.requestId ? { request_id: request.requestId } : {}),
    session_id: request.sessionId ?? undefined
  }

  try {
    await requestForOwnedSession(
      request.sessionId,
      ambientRequestFor(gateway),
      'approval.respond',
      params,
      APPROVAL_RESPOND_REQUEST_TIMEOUT_MS
    )
  } catch (error) {
    if (isRequestTimeoutError(error) && !request.requestId) {
      // The answer may have landed; a stale legacy prompt must not be resubmitted
      // against a newer FIFO entry. Retire it (and only it: an identified prompt that arrived
      // meanwhile keeps its server route), then restore the server's pending state.
      clearApprovalRequest(request.sessionId, undefined, item => !item.requestId)
      void replayPendingApproval(gateway, request.sessionId).catch(() => undefined)
      throw error
    }

    if (!isRequestTimeoutError(error)) {
      throw error
    }

    // The deadline fired while the approval may still be pending server-side
    // (WS stall behind a long LLM stream). Resolve by request_id is idempotent: re-send once.
    await requestForOwnedSession(
      request.sessionId,
      ambientRequestFor(gateway),
      'approval.respond',
      params,
      APPROVAL_RESPOND_REQUEST_TIMEOUT_MS
    )
  }
}

/** The prompt request for one specific session — the tile counterpart of the
 *  active-session `$*Request` views (same map, fixed key). */
export const sessionApprovalStackSize = (sessionId: string | null) =>
  computed($approvalStackSizes, sizes => sizes[keyFor(sessionId)] ?? 0)
export const sessionApprovalRequests = (sessionId: string | null) =>
  computed($approvalQueues, all => all[keyFor(sessionId)] ?? EMPTY_APPROVALS)
export const sessionApprovalRequest = (sessionId: string | null) =>
  computed(approval.$all, all => all[keyFor(sessionId)] ?? null)
/** True while the session holds a command approval. Folder and site cards are
 *  not about the turn's tool rows, so they never stand in for them. */
export const sessionCommandApprovalPending = (sessionId: string | null) =>
  computed($approvalQueues, all => (all[keyFor(sessionId)] ?? EMPTY_APPROVALS).some(isCommandApproval))
/** A session's sudo card, else the app-level one (a Bot Screen package install is raised with no
 *  session: it belongs to the connection, not to a turn, so whichever chat is focused shows it). */
export const sessionSudoRequest = (sessionId: string | null) =>
  computed(sudo.$all, all => all[keyFor(sessionId)] ?? (sessionId ? (all[keyFor(null)] ?? null) : null))
export const sessionSecretRequest = (sessionId: string | null) =>
  computed(secret.$all, all => all[keyFor(sessionId)] ?? null)

export const $sudoRequest = sudo.$active
export const $sudoRequests = sudo.$all
export const setSudoRequest = sudo.set
export const clearSudoRequest = sudo.clear

export const $secretRequest = secret.$active
export const $secretRequests = secret.$all
export const setSecretRequest = secret.set
export const clearSecretRequest = secret.clear

export const $vaultUnlockRequest = vaultUnlock.$active
export const setVaultUnlockRequest = vaultUnlock.set
export const clearVaultUnlockRequest = vaultUnlock.clear
export const $vaultUnlockRequests = vaultUnlock.$all
export const sessionVaultUnlockRequest = (sessionId: string | null) =>
  computed(vaultUnlock.$all, all => all[keyFor(sessionId)] ?? null)

export const $vaultSaveLoginRequest = vaultSave.$active
export const setVaultSaveLoginRequest = vaultSave.set
export const clearVaultSaveLoginRequest = vaultSave.clear
export const $vaultSaveLoginRequests = vaultSave.$all
export const sessionVaultSaveLoginRequest = (sessionId: string | null) =>
  computed(vaultSave.$all, all => all[keyFor(sessionId)] ?? null)

export const $vaultCodeRequest = vaultCode.$active
export const setVaultCodeRequest = vaultCode.set
export const clearVaultCodeRequest = vaultCode.clear
export const $vaultCodeRequests = vaultCode.$all
export const sessionVaultCodeRequest = (sessionId: string | null) =>
  computed(vaultCode.$all, all => all[keyFor(sessionId)] ?? null)

// True when the active session is blocked on the user (clarify question or an
// approval / sudo / secret prompt). Mirrors the pet's `awaitingInput` concept
// (agent/pet/state.py): the turn is paused on you, not working — so callers can
// suppress "thinking" indicators and the Esc-to-interrupt shortcut while you
// decide, instead of treating the wait as an in-flight turn.
export const $activeSessionAwaitingInput = computed(
  [
    $clarifyRequest,
    $approvalRequest,
    $sudoRequest,
    $secretRequest,
    $vaultUnlockRequest,
    $vaultSaveLoginRequest,
    $vaultCodeRequest
  ],
  (clarify, approval, sudo, secret, vault, save, code) =>
    Boolean(clarify || approval || sudo || secret || vault || save || code)
)

/** True when `sessionId` is parked on a blocking prompt that typing cannot
 *  answer (approval / sudo / secret). Clarify is deliberately excluded: typed
 *  text answers a clarify's next question (the composer locks it through
 *  `answerParkedCard`), but no message text can approve a command or supply a
 *  password. Imperative read — the composer checks this on Enter,
 *  not on every render. */
export const hasBlockingPromptRequest = (sessionId: string | null | undefined): boolean => {
  const key = keyFor(sessionId)

  return Boolean(
    approval.$all.get()[key] ||
    sudo.$all.get()[key] ||
    secret.$all.get()[key] ||
    vaultUnlock.$all.get()[key] ||
    vaultSave.$all.get()[key] ||
    vaultCode.$all.get()[key]
  )
}

/** Reactive twin of `hasBlockingPromptRequest`, for the composer's busy-action
 *  affordance (the primary button must advertise queue, not steer, while the
 *  turn is parked on a prompt Enter can't answer). */
export const sessionBlockingPrompt = (sessionId: string | null) =>
  computed(
    [approval.$all, sudo.$all, secret.$all, vaultUnlock.$all, vaultSave.$all, vaultCode.$all],
    (approvals, sudos, secrets, vaults, saves, codes) => {
      const key = keyFor(sessionId)

      return Boolean(approvals[key] || sudos[key] || secrets[key] || vaults[key] || saves[key] || codes[key])
    }
  )

/** Per-session `awaitingInput` — the tile composer's counterpart of
 *  `$activeSessionAwaitingInput` (same sources, fixed session instead of the
 *  active one). */
export function sessionAwaitingInput(sessionId: string | null) {
  return computed(
    [$clarifyRequests, approval.$all, sudo.$all, secret.$all, vaultUnlock.$all, vaultSave.$all, vaultCode.$all],
    (clarify, approvals, sudos, secrets, vaults, saves, codes) => {
      const key = keyFor(sessionId)

      return Boolean(
        clarify[key] || approvals[key] || sudos[key] || secrets[key] || vaults[key] || saves[key] || codes[key]
      )
    }
  )
}

// Drop in-flight prompts for `sessionId` across every kind — or every parked
// prompt when no session is given (global reset / tests). `turn-end` is for a
// turn that ended on its own (complete, error, running→false): folder and site
// approvals outlive it because the sandbox, not the turn, waits on them. Stop,
// close and reclaim use `all`.
export type ClearPromptsMode = 'all' | 'turn-end'

export function clearAllPrompts(sessionId?: string | null, mode: ClearPromptsMode = 'all'): void {
  if (sessionId === undefined) {
    approval.reset()
    sudo.reset()
    secret.reset()
    vaultUnlock.reset()
    vaultSave.reset()
    vaultCode.reset()

    return
  }

  approval.clear(sessionId, undefined, mode === 'turn-end' ? isCommandApproval : undefined)
  sudo.clear(sessionId)
  secret.clear(sessionId)
  vaultUnlock.clear(sessionId)
  vaultSave.clear(sessionId)
  vaultCode.clear(sessionId)
}
