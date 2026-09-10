import { $activeSessionId } from '@/store/session'
import { $focusedRuntimeId, $sessionTiles } from '@/store/session-states'

/** A session the user can see: focused runtime, primary active chat, or an
 *  open session tile. Preview open/close/act share this so a focused tile
 *  can drive the in-app browser without the primary-only `isActiveEvent`
 *  gate rejecting the turn. Hidden background sessions stay blocked. */
export function sessionIsOnScreen(sessionId: string): boolean {
  if (!sessionId) {
    return false
  }

  return (
    sessionId === $focusedRuntimeId.get() ||
    sessionId === $activeSessionId.get() ||
    $sessionTiles.get().some(tile => tile.runtimeId === sessionId)
  )
}
