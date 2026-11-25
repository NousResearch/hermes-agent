import type { MessagingPlatformUpdateResponse } from '@/hermes'

/** Whether the running gateway applied a save itself. A backend that sends `live` answers for every
 *  profile; an older one only knows `hot_served` (multiplexed named profiles). */
export function appliedLive(result: MessagingPlatformUpdateResponse): boolean {
  return result.live ? result.live.applied : Boolean(result.hot_served)
}

/** Card state to show right after an enable/disable toggle, before the next status read. */
export function optimisticToggleState(enabled: boolean, configured: boolean, live: boolean): string {
  if (!enabled) {
    return 'disabled'
  }

  if (!configured) {
    return 'not_configured'
  }

  return live ? 'connecting' : 'pending_restart'
}

/** Toast line for a save the gateway could not apply live: its reason when it gave one. */
export function restartReason(result: MessagingPlatformUpdateResponse, fallback: string): string {
  const detail = result.live?.applied === false ? result.live.detail : undefined

  return detail ? `${fallback} (${detail})` : fallback
}
