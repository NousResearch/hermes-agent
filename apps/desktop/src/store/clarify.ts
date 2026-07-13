import { atom, computed, type ReadableAtom } from 'nanostores'

import { $activeGatewayProfile } from './profile'
import {
  promptIdentityKey,
  type PromptRequestInput,
  type PromptTarget,
  stampPromptRequest,
  type StoredPromptRequest
} from './prompt-identity'
import { $activeSessionId } from './session'

export interface ClarifyRequest {
  profile: string
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
}

const $allClarifyRequests = atom<Record<string, StoredPromptRequest<ClarifyRequest>>>({})
export const $clarifyRequests: ReadableAtom<Record<string, StoredPromptRequest<ClarifyRequest>>> = $allClarifyRequests

// The clarify request for the currently-viewed session. The inline ClarifyTool
// only ever mounts inside the active session's transcript, so it reads this
// focus-scoped view rather than reaching into the whole map.
export const $clarifyRequest = computed(
  [$clarifyRequests, $activeGatewayProfile, $activeSessionId],
  (requests, activeProfile, activeId) => requests[promptIdentityKey(activeProfile, activeId)] ?? null
)

export function setClarifyRequest(request: PromptRequestInput<ClarifyRequest>): void {
  const stamped = stampPromptRequest('clarify', request)
  $allClarifyRequests.set({
    ...$allClarifyRequests.get(),
    [promptIdentityKey(stamped.profile, stamped.sessionId)]: stamped
  })
}

export function clearClarifyRequest(target?: PromptTarget | string): void {
  const requests = $allClarifyRequests.get()

  if (typeof target === 'object') {
    const key = promptIdentityKey(target.profile, target.sessionId)
    const current = requests[key]

    if (!current || (target.requestId && current.requestId !== target.requestId)) {
      return
    }

    const next = { ...requests }
    delete next[key]
    $allClarifyRequests.set(next)

    return
  }

  // Preserve the global reset and legacy request-id sweep used by teardown.
  const next: Record<string, StoredPromptRequest<ClarifyRequest>> = {}
  let changed = false

  for (const [key, value] of Object.entries(requests)) {
    if (target && value.requestId !== target) {
      next[key] = value
    } else {
      changed = true
    }
  }

  if (changed) {
    $allClarifyRequests.set(next)
  }
}
