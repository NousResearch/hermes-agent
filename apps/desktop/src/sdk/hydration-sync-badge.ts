import { $activeGatewayProfile, $hydrationSyncProfile, normalizeProfileKey } from '@/store/profile'

/** Ceiling on the paint-first "Syncing…" badge. The profile gate it waits on is
 *  not guaranteed to ever fire (see beginHydrationBackgroundSync), so the badge
 *  needs a bound of its own or it outlives the wake it describes. */
export const HYDRATION_SYNC_BADGE_TIMEOUT_MS = 30_000

// Raise the "Syncing…" affordance for a paint-first wake (#89843) and tear it
// down as soon as the active-profile gate catches up. The listener clears ONLY
// its own profile's badge: a newer wake may have replaced the badge with a
// different profile, and the stale listener must not wipe the winner's.
//
// The gate is not guaranteed to fire. `.listen()` is change-only, and a wake
// is routed here precisely because $activeGatewayProfile did not match at
// resolve time. ensureGatewayProfile does publish the target on a
// shared-primary connection, but when the activation did NOT land it publishes
// the route the registry actually settled on instead — so on that path the
// atom may never become this profile and the listener never fires. Without a
// cap the badge would outlive the wake it describes and strand a permanent
// "Syncing <profile>…" spinner with no user-reachable dismissal; only a full
// app restart would clear it. Cap the wait so the badge can never outlive the
// work.
export function beginHydrationBackgroundSync(profile: string): void {
  $hydrationSyncProfile.set(profile)

  let timer: number | undefined

  const clearOwnBadge = (): void => {
    if ($hydrationSyncProfile.get() === profile) {
      $hydrationSyncProfile.set(null)
    }
  }

  const unlisten = $activeGatewayProfile.listen(next => {
    if (normalizeProfileKey(next) === profile) {
      clearOwnBadge()

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }

      unlisten()
    }
  })

  timer = window.setTimeout(() => {
    clearOwnBadge()
    unlisten()
  }, HYDRATION_SYNC_BADGE_TIMEOUT_MS)
}
