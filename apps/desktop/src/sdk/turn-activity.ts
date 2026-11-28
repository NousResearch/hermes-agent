import { computed } from 'nanostores'

import { $sessionStates } from '@/store/session-states'

/** Runtime session id → mid-turn. Not gateway socket state. */
export const $busyBySession = computed($sessionStates, states => {
  const map: Record<string, boolean> = {}

  for (const [id, state] of Object.entries(states)) {
    map[id] = Boolean(state.busy)
  }

  return map
})
