import { useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'

/**
 * Notifications settings' plugin seam — a render area at the END of the
 * alerts section, so a plugin that raises its own alerts (Kanban's alerts
 * mode) puts its preference next to the native-notification kinds instead of
 * burying it in a plugin page.
 */
export const NOTIFICATIONS_AREAS = {
  /** Appended to Settings → Notifications → Alerts, after the built-in rows. */
  extra: 'notifications.extra'
} as const

/** Mounts every `notifications.extra` registration (own error boundary each,
 *  so a broken contribution degrades to an inline error instead of a dead page). */
export function NotificationsExtraSlot() {
  const contributions = useContributions(NOTIFICATIONS_AREAS.extra)

  if (contributions.length === 0) {
    return null
  }

  return (
    <>
      {contributions.map(contribution => (
        <ContribBoundary id={contribution.id} key={`${contribution.source ?? 'core'}:${contribution.id}`}>
          {contribution.render && <ContribRender render={contribution.render} />}
        </ContribBoundary>
      ))}
    </>
  )
}
