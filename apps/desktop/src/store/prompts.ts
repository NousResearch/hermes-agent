import { atom, computed, type ReadableAtom } from 'nanostores'

import { $clarifyRequest } from './clarify'
import { $activeGatewayProfile } from './profile'
import {
  type PromptIdentity,
  promptIdentityKey,
  type PromptRequestInput,
  type PromptTarget,
  stampPromptRequest,
  type StoredPromptRequest
} from './prompt-identity'
import { $activeSessionId } from './session'

// Blocking interactive prompts the gateway raises mid-turn. Each maps to a
// `*.request` event the Python side emits while it blocks the agent thread
// waiting for a `*.respond` RPC. Without a renderer for these, the agent
// silently stalls until its timeout (default 5 min) and the tool is BLOCKED.
//
// Like clarify, every prompt is parked under the normalized profile and runtime
// session that raised it. The exported $*Request view is scoped to both active
// identities, so a background prompt never hijacks the foreground.

interface PromptStore<T extends PromptIdentity> {
  $active: ReadableAtom<null | StoredPromptRequest<T>>
  $all: ReadableAtom<Record<string, StoredPromptRequest<T>>>
  clear: (target?: PromptTarget) => void
  reset: () => void
  set: (request: PromptRequestInput<T>) => void
}

// One per-profile/session prompt kind, plus an active-identity view for the
// overlays. A request-id mismatch is a no-op so a stale resolve cannot wipe a
// replacement. Calling clear without a target preserves the global reset path.
function keyedPromptStore<T extends PromptIdentity>(kind: string): PromptStore<T> {
  const $all = atom<Record<string, StoredPromptRequest<T>>>({})
  const idOf = (value: StoredPromptRequest<T>): string | undefined => (value as { requestId?: string }).requestId

  return {
    $active: computed(
      [$all, $activeGatewayProfile, $activeSessionId],
      (all, activeProfile, activeId) => all[promptIdentityKey(activeProfile, activeId)] ?? null
    ),
    $all,
    reset: () => $all.set({}),
    set(request) {
      const stamped = stampPromptRequest(kind, request)
      $all.set({ ...$all.get(), [promptIdentityKey(stamped.profile, stamped.sessionId)]: stamped })
    },
    clear(target) {
      if (!target) {
        $all.set({})

        return
      }

      const all = $all.get()
      const key = promptIdentityKey(target.profile, target.sessionId)
      const current = all[key]

      if (!current || (target.requestId && idOf(current) !== target.requestId)) {
        return
      }

      const next = { ...all }
      delete next[key]
      $all.set(next)
    }
  }
}

// Approval is session-keyed on the backend (one in-flight approval per session,
// resolved via approval.respond {choice, session_id}). It carries no request_id,
// unlike sudo/secret which are _block()-style request/response.
export interface ApprovalRequest extends PromptIdentity {
  // false when the backend won't honor a permanent allow (tirith warning) → hide "Always allow".
  allowPermanent?: boolean
  choices?: string[]
  command: string
  description: string
  smartDenied?: boolean
}

export interface SudoRequest extends PromptIdentity {
  requestId: string
}

export interface SecretRequest extends PromptIdentity {
  envVar: string
  prompt: string
  requestId: string
}

const approval = keyedPromptStore<ApprovalRequest>('approval')
const sudo = keyedPromptStore<SudoRequest>('sudo')
const secret = keyedPromptStore<SecretRequest>('secret')
const $approvalInlineAnchorCount = atom(0)

export const $approvalRequest = approval.$active
export const $approvalRequests = approval.$all
export const setApprovalRequest = approval.set
export const clearApprovalRequest = approval.clear
export const $approvalInlineVisible = computed($approvalInlineAnchorCount, count => count > 0)

export function registerApprovalInlineAnchor(): () => void {
  $approvalInlineAnchorCount.set($approvalInlineAnchorCount.get() + 1)

  return () => {
    $approvalInlineAnchorCount.set(Math.max(0, $approvalInlineAnchorCount.get() - 1))
  }
}

export const $sudoRequest = sudo.$active
export const $sudoRequests = sudo.$all
export const setSudoRequest = sudo.set
export const clearSudoRequest = sudo.clear

export const $secretRequest = secret.$active
export const $secretRequests = secret.$all
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

// Drop in-flight prompts for one exact profile/session identity across all three
// kinds — or every parked prompt when no target is given (global reset/tests).
export function clearAllPrompts(target?: PromptIdentity): void {
  if (!target) {
    approval.reset()
    sudo.reset()
    secret.reset()
    $approvalInlineAnchorCount.set(0)

    return
  }

  approval.clear(target)
  sudo.clear(target)
  secret.clear(target)
}
