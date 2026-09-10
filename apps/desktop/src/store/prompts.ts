import { atom, computed, type ReadableAtom } from 'nanostores'

import { $clarifyRequest, $clarifyRequests } from './clarify'
import { isSessionGone, isSessionGoneForBackgroundPolling, markSessionGone } from './runtime-gone'
import { $activeSessionId } from './session'
import { ambientRequestFor } from './session-gone-latch'
import { requestForOwnedSession } from './session-states'

// Blocking interactive prompts the gateway raises mid-turn. Each maps to a
// `*.request` event the Python side emits while it blocks the agent thread
// waiting for a `*.respond` RPC. Without a renderer for these, the agent
// silently stalls until its timeout (default 5 min) and the tool is BLOCKED.
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
    $active: computed([$all, $activeSessionId], (all, activeId) => all[keyFor(activeId)] ?? null),
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

// Approval is session-keyed on the backend and correlated by `request_id` when
// available (legacy ID-free responses remain FIFO-compatible). Resolved via
// approval.respond {choice, request_id, session_id}.
export interface ApprovalRequest extends KeyedPrompt {
  // false when the backend won't honor a permanent allow (tirith warning) → hide "Always allow".
  allowPermanent?: boolean
  choices?: string[]
  command: string
  description: string
  requestId?: string
  smartDenied?: boolean
}

interface ApprovalGateway {
  request: (method: string, params: Record<string, unknown>) => Promise<unknown>
}

interface PendingApprovalPayload {
  allow_permanent?: boolean
  choices?: unknown
  command?: unknown
  description?: unknown
  request_id?: unknown
  smart_denied?: boolean
}

export interface SudoRequest extends KeyedPrompt {
  requestId: string
}

export interface SecretRequest extends KeyedPrompt {
  envVar: string
  prompt: string
  requestId: string
}

const EMPTY_APPROVALS: ApprovalRequest[] = []
const $approvalQueues = atom<Record<string, ApprovalRequest[]>>({})
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
    $approvalQueues.set({})
  },
  set(request: ApprovalRequest) {
    const key = keyFor(request.sessionId)
    const queues = $approvalQueues.get()
    const queue = queues[key] ?? EMPTY_APPROVALS
    const index = queue.findIndex(item => item.requestId === request.requestId)
    const next = [...queue]

    if (index < 0) {
      next.push(request)
    } else {
      next[index] = request
    }

    $approvalQueues.set({ ...queues, [key]: next })
  },
  clear(sessionId?: string | null, requestId?: string) {
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

      const remaining = requestId ? queue.filter(item => item.requestId !== requestId) : EMPTY_APPROVALS

      if (remaining.length === queue.length) {
        continue
      }

      changed = true
      if (remaining.length) {
        next[key] = remaining
      } else {
        delete next[key]
      }
    }

    if (changed) {
      $approvalQueues.set(next)
    }
  }
}
const sudo = keyedPromptStore<SudoRequest>()
const secret = keyedPromptStore<SecretRequest>()

export const $approvalRequest = computed([approval.$all, $activeSessionId], (all, activeId) => all[keyFor(activeId)] ?? null)
export const setApprovalRequest = approval.set
export const clearApprovalRequest = approval.clear

export async function receiveApprovalRequest(gateway: ApprovalGateway | null, request: ApprovalRequest): Promise<void> {
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

  await Promise.all(result.approvals.map(pending => {
    if (typeof pending.request_id !== 'string') {
      return
    }

    return receiveApprovalRequest(gateway, {
      allowPermanent: pending.allow_permanent !== false,
      choices: Array.isArray(pending.choices) ? pending.choices.filter(choice => typeof choice === 'string') : undefined,
      command: typeof pending.command === 'string' ? pending.command : '',
      description: typeof pending.description === 'string' ? pending.description : 'dangerous command',
      requestId: pending.request_id,
      sessionId,
      smartDenied: pending.smart_denied === true
    })
  }))
}

/** The prompt request for one specific session — the tile counterpart of the
 *  active-session `$*Request` views (same map, fixed key). */
export const sessionApprovalRequests = (sessionId: string | null) =>
  computed($approvalQueues, all => all[keyFor(sessionId)] ?? EMPTY_APPROVALS)
export const sessionApprovalRequest = (sessionId: string | null) =>
  computed(approval.$all, all => all[keyFor(sessionId)] ?? null)
export const sessionSudoRequest = (sessionId: string | null) =>
  computed(sudo.$all, all => all[keyFor(sessionId)] ?? null)
export const sessionSecretRequest = (sessionId: string | null) =>
  computed(secret.$all, all => all[keyFor(sessionId)] ?? null)

export const $sudoRequest = sudo.$active
export const setSudoRequest = sudo.set
export const clearSudoRequest = sudo.clear

export const $secretRequest = secret.$active
export const setSecretRequest = secret.set
export const clearSecretRequest = secret.clear

// True when the active session is blocked on the user (clarify question or an
// approval / sudo / secret prompt). Mirrors the pet's `awaitingInput` concept
// (agent/pet/state.py): the turn is paused on you, not working — so callers can
// suppress "thinking" indicators and the Esc-to-interrupt shortcut while you
// decide, instead of treating the wait as an in-flight turn.
export const $activeSessionAwaitingInput = computed(
  [$clarifyRequest, $approvalRequest, $sudoRequest, $secretRequest],
  (clarify, approval, sudo, secret) => Boolean(clarify || approval || sudo || secret)
)

/** True when `sessionId` is parked on a blocking prompt that typing cannot
 *  answer (approval / sudo / secret). Clarify is deliberately excluded: typing
 *  a real message IS an answer to a clarify ("none of these" — the composer
 *  skips it and routes the words), but no message text can approve a command
 *  or supply a password. Imperative read — the composer checks this on Enter,
 *  not on every render. */
export const hasBlockingPromptRequest = (sessionId: string | null | undefined): boolean => {
  const key = keyFor(sessionId)

  return Boolean(approval.$all.get()[key] || sudo.$all.get()[key] || secret.$all.get()[key])
}

/** Reactive twin of `hasBlockingPromptRequest`, for the composer's busy-action
 *  affordance (the primary button must advertise queue, not steer, while the
 *  turn is parked on a prompt Enter can't answer). */
export const sessionBlockingPrompt = (sessionId: string | null) =>
  computed([approval.$all, sudo.$all, secret.$all], (approvals, sudos, secrets) => {
    const key = keyFor(sessionId)

    return Boolean(approvals[key] || sudos[key] || secrets[key])
  })

/** Per-session `awaitingInput` — the tile composer's counterpart of
 *  `$activeSessionAwaitingInput` (same sources, fixed session instead of the
 *  active one). */
export function sessionAwaitingInput(sessionId: string | null) {
  return computed([$clarifyRequests, approval.$all, sudo.$all, secret.$all], (clarify, approvals, sudos, secrets) => {
    const key = keyFor(sessionId)

    return Boolean(clarify[key] || approvals[key] || sudos[key] || secrets[key])
  })
}

// Drop in-flight prompts for `sessionId` (a turn ended) across all three kinds —
// or every parked prompt when no session is given (global reset / tests).
export function clearAllPrompts(sessionId?: string | null): void {
  if (sessionId === undefined) {
    approval.reset()
    sudo.reset()
    secret.reset()

    return
  }

  approval.clear(sessionId)
  sudo.clear(sessionId)
  secret.clear(sessionId)
}
