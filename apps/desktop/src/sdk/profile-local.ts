// `host.warmProfile` and `host.dropLeftoverProfilePrefs`: profile verbs that
// touch only this window's pool and storage, never the profile itself.

import { dropBrowserProfilePrefs } from '@/lib/browser-profile-prefs'
import { prewarmProfileBackend } from '@/store/profile'

/** Pre-dial a profile's gateway socket in the background — pool-only, no
 *  activation, no navigation, no scope change. Delegates to
 *  prewarmProfileBackend so plugin surfaces get the SAME pool-saturation
 *  guard, hover dwell, and per-profile throttle as the built-in rail
 *  (#91545): a pointer sweep across a plugin roster (bot-row's
 *  onPointerEnter fires with no dwell of its own) previously spawned at
 *  pointer speed, filled the local backend pool past maxBackends, and left
 *  the next profile's spawn queued until the 30s slot timeout — observed
 *  as a profile surface that hangs forever while every other profile
 *  renders. It already no-ops for shared-remote routes and the primary.
 *  Fire-and-forget: failures are swallowed — the click path re-runs its
 *  own ensure and surfaces errors properly. */
export function warmProfile(profile: string): void {
  const name = (profile ?? '').trim()

  if (!name) {
    return
  }

  prewarmProfileBackend(name)
}

/** Call after a plugin's own `profiles.create` succeeded, never before: a
 *  create never lands on a live profile's name, so a folder or startup
 *  profile this tab stored under it was left by a deleted profile. The same
 *  cleanup the core create dialog runs. A profile created on another
 *  connection (`connectionId` neither empty nor `local`) is not this tab's,
 *  so the name here keeps its prefs. No-op outside the browser-hosted app. */
export function dropLeftoverProfilePrefs(profile: string, connectionId?: null | string): void {
  const connection = String(connectionId ?? '').trim()

  if (connection && connection !== 'local') {
    return
  }

  dropBrowserProfilePrefs(profile)
}
