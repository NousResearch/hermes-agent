import type { ClientSessionState } from '@/app/types'
import type { GatewayEventPayload } from '@/lib/chat-messages'

export type TurnStateTransition = 'settle' | 'snapshot' | 'start'

export interface TurnStateReconciliation {
  accepted: boolean
  state: ClientSessionState
}

export function reconcileClientTurnState(
  state: ClientSessionState,
  payload: GatewayEventPayload | undefined,
  transition: TurnStateTransition
): TurnStateReconciliation {
  const rawGeneration = payload?.turn_generation

  const generation =
    typeof rawGeneration === 'number' && Number.isSafeInteger(rawGeneration) && rawGeneration >= 0
      ? rawGeneration
      : null

  const rawRevision = payload?.turn_state_revision

  const revision =
    typeof rawRevision === 'number' && Number.isSafeInteger(rawRevision) && rawRevision >= 0 ? rawRevision : null

  const running =
    transition === 'start'
      ? true
      : transition === 'settle'
        ? false
        : typeof payload?.running === 'boolean'
          ? payload.running
          : undefined

  if (revision !== null) {
    if (revision < state.turnStateRevision) {
      return { accepted: false, state }
    }

    if (revision === state.turnStateRevision) {
      if (generation !== null && generation < state.turnGeneration) {
        return { accepted: false, state }
      }

      // A terminal transition dominates an active snapshot at the same fence.
      // New backends advance the revision on settle, so this is principally a
      // duplicate/out-of-order defense. Missing revisions intentionally retain
      // the legacy generation-only behavior below.
      if (running === true && state.busy === false && revision > 0) {
        return { accepted: false, state }
      }
    }
  } else if (generation !== null && generation < state.turnGeneration) {
    return { accepted: false, state }
  }

  let turnOrigin = state.turnOrigin

  if (payload && Object.hasOwn(payload, 'turn_origin')) {
    turnOrigin =
      payload.turn_origin === 'user' || payload.turn_origin === 'notification' || payload.turn_origin === 'goal'
        ? payload.turn_origin
        : null
  } else if (transition === 'start') {
    turnOrigin = 'user'
  }

  return {
    accepted: true,
    state: {
      ...state,
      ...(running === undefined ? {} : { busy: running }),
      ...(generation === null ? {} : { turnGeneration: generation }),
      ...(revision === null ? {} : { turnStateRevision: revision }),
      turnOrigin
    }
  }
}
