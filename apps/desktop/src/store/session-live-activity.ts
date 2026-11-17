import { atom } from 'nanostores'

import { normalizeProfileKey } from './profile'

// What each working turn is doing right now ("executing tool: terminal"), per
// profile and STORED session id, from the `session.active_list` snapshot. The
// session list carries the same durable stamp, but the list is not re-read on
// an activity heartbeat (no `sessions.changed` fires for one), so a row reading
// only the list would freeze on its turn's first label. Each profile's poll
// replaces only its own bucket; stored ids can repeat across profiles, so a
// row reads its OWN profile's bucket. Gateway-bound: a switch clears it.

export type LiveActivityByProfile = Readonly<Record<string, Readonly<Record<string, string>>>>

export const $liveActivityBySession = atom<LiveActivityByProfile>({})

export function setLiveActivity(profile: string, activity: Record<string, string>): void {
  const key = normalizeProfileKey(profile)
  const current = $liveActivityBySession.get()
  const previous = current[key] ?? {}
  const ids = Object.keys(activity)

  if (ids.length === Object.keys(previous).length && ids.every(id => previous[id] === activity[id])) {
    return
  }

  $liveActivityBySession.set({ ...current, [key]: activity })
}

export function resetLiveActivity(): void {
  $liveActivityBySession.set({})
}

/** The live label for a row: its own profile's bucket, checked under every id
 *  of its compression lineage (the snapshot keys a compressed conversation by
 *  its logical owner, the sidebar row by its continuation). */
export function liveActivityFor(
  byProfile: LiveActivityByProfile,
  profile: string | undefined,
  ids: readonly string[]
): string | undefined {
  const bucket = byProfile[normalizeProfileKey(profile)]

  return bucket ? ids.map(id => bucket[id]).find(Boolean) : undefined
}
