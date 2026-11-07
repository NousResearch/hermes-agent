import { $simplifiedAgentMode } from './simplified-agent-mode'

export const SIMPLIFIED_NEW_SESSION_EVENT = 'hermes:simplified-new-session'

/** A mounted v2 sidebar claims the action; otherwise the existing path runs. */
export function requestSimplifiedNewSession(): boolean {
  return (
    $simplifiedAgentMode.get() && !window.dispatchEvent(new Event(SIMPLIFIED_NEW_SESSION_EVENT, { cancelable: true }))
  )
}
