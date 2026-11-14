import { $activeSessionId, $selectedStoredSessionId } from './session'
import { $sessionStates, $sessionTiles } from './session-states'

// Shared by row actions: never substitute an unrelated foreground conversation.
export function runtimeIdForStoredSession(storedSessionId: string): null | string {
  if (storedSessionId === $selectedStoredSessionId.get()) {
    const active = $activeSessionId.get()

    if (active) {
      return active
    }
  }

  const tileRuntimeId = $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)?.runtimeId

  if (tileRuntimeId) {
    return tileRuntimeId
  }

  for (const [runtimeId, state] of Object.entries($sessionStates.get())) {
    if (state.storedSessionId === storedSessionId) {
      return runtimeId
    }
  }

  return null
}
