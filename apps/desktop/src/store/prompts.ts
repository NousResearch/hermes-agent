import { atom, computed, type ReadableAtom } from 'nanostores'

import { $clarifyRequest, $clarifyRequests } from './clarify'
import { isSessionGone, isSessionGoneForBackgroundPolling, markSessionGone } from './runtime-gone'
import { respondToServerRequest } from './server-requests'
import { $activeSessionId } from './session'
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

// Approval is queue-backed on the backend (`tools/approval.py`): `requestId` is
// the QUEUE entry's id (what `approval.pending` / `approval.received` /
// `approval.respond` key on), stable across delivery paths. The live prompt
// arrives as an `approval` server request whose id is `serverRequestId`; the
// card answers that request when it is still open and falls back to the
// `approval.respond` RPC when the prompt was restored from `approval.pending`.
export interface ApprovalRequest extends KeyedPrompt {
  // false when the backend won't honor a permanent allow (tirith warning) → hide "Always allow".
  allowPermanent?: boolean
  choices?: string[]
  command: string
  description: string
  requestId?: string
  serverRequestId?: string
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
  command?: string
  requestId: string
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
export const $approvalRequest = computed([approval.$all, $activeSessionId], (all, activeId) => all[keyFor(activeId)] ?? null)
export const setApprovalRequest = approval.set
export const clearApprovalRequest = approval.clear

export async function receiveApprovalRequest(gateway: ApprovalGateway | null, request: ApprovalRequest): Promise<void> {
  // A prompt restored from `approval.pending` must not clobber the live server
  // request that already carries the same queue entry (it knows how to answer).
  const current = $approvalQueues.get()[keyFor(request.sessionId)]?.find(item => item.requestId === request.requestId)

  if (current?.requestId && current.requestId === request.requestId && current.serverRequestId && !request.serverRequestId) {
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
        smartDenied: pending.smart_denied === true
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
export async function answerApproval(
  gateway: ApprovalGateway | null,
  request: Pick<ApprovalRequest, 'requestId' | 'serverRequestId' | 'sessionId'>,
  choice: string,
  all = false
): Promise<void> {
  if (respondToServerRequest(request.serverRequestId, { choice, ...(all ? { all: true } : {}) })) {
    return
  }

  if (!gateway) {
    throw new Error('Hermes gateway is not connected')
  }

  await requestForOwnedSession(request.sessionId, ambientRequestFor(gateway), 'approval.respond', {
    all,
    choice,
    ...(request.requestId ? { request_id: request.requestId } : {}),
    session_id: request.sessionId ?? undefined
  })
}

/** The prompt request for one specific session — the tile counterpart of the
 *  active-session `$*Request` views (same map, fixed key). */
export const sessionApprovalStackSize = (sessionId: string | null) =>
  computed($approvalStackSizes, sizes => sizes[keyFor(sessionId)] ?? 0)
export const sessionApprovalRequests = (sessionId: string | null) =>
  computed($approvalQueues, all => all[keyFor(sessionId)] ?? EMPTY_APPROVALS)
export const sessionApprovalRequest = (sessionId: string | null) =>
  computed(approval.$all, all => all[keyFor(sessionId)] ?? null)
export const sessionSudoRequest = (sessionId: string | null) =>
  computed(sudo.$all, all => all[keyFor(sessionId)] ?? null)
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
 *  answer (approval / sudo / secret). Clarify is deliberately excluded: typing
 *  a real message IS an answer to a clarify ("none of these" — the composer
 *  skips it and routes the words), but no message text can approve a command
 *  or supply a password. Imperative read — the composer checks this on Enter,
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

// Drop in-flight prompts for `sessionId` (a turn ended) across all three kinds —
// or every parked prompt when no session is given (global reset / tests).
export function clearAllPrompts(sessionId?: string | null): void {
  if (sessionId === undefined) {
    approval.reset()
    sudo.reset()
    secret.reset()
    vaultUnlock.reset()
    vaultSave.reset()
    vaultCode.reset()

    return
  }

  approval.clear(sessionId)
  sudo.clear(sessionId)
  secret.clear(sessionId)
  vaultUnlock.clear(sessionId)
  vaultSave.clear(sessionId)
  vaultCode.clear(sessionId)
}
