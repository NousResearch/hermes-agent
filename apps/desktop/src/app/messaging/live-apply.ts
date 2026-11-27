import type { MessagingPlatformUpdateResponse } from '@/hermes'

/** Whether the running gateway applied a save itself. A backend that sends `live` answers for every
 *  profile; an older one only knows `hot_served` (multiplexed named profiles). */
export function appliedLive(result: MessagingPlatformUpdateResponse): boolean {
  return result.live ? result.live.applied : Boolean(result.hot_served)
}

/** The gateway's verdict that the platform is enabled but lacks the credentials it needs: a restart
 *  would not start it either, so the card says "needs setup", never "restart". */
export function needsCredentials(result: MessagingPlatformUpdateResponse): boolean {
  return result.live?.state === 'not_configured'
}

/** Card state to show right after an enable/disable toggle, before the next status read. */
export function optimisticToggleState(
  enabled: boolean,
  configured: boolean,
  result: MessagingPlatformUpdateResponse
): string {
  if (!enabled) {
    return 'disabled'
  }

  if (!configured || needsCredentials(result)) {
    return 'not_configured'
  }

  return appliedLive(result) ? 'connecting' : 'pending_restart'
}

/** Platforms whose latest save the running gateway could not apply: only these keep the restart banner
 *  up, and a later save it applied for the same platform takes that platform off. */
export function restartPendingAfter(
  pending: ReadonlySet<string>,
  platformId: string,
  result: MessagingPlatformUpdateResponse
): ReadonlySet<string> {
  const next = new Set(pending)

  if (appliedLive(result)) {
    next.delete(platformId)
  } else {
    next.add(platformId)
  }

  return next
}

/** Toast line for a save the gateway could not apply live: its reason when it gave one. */
export function restartReason(result: MessagingPlatformUpdateResponse, fallback: string): string {
  const detail = result.live?.applied === false ? result.live.detail : undefined

  return detail ? `${fallback} (${detail})` : fallback
}
