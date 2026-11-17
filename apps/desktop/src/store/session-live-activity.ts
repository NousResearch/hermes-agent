import { atom } from 'nanostores'

import { stableRecord } from '@/lib/stable-array'

// What each working turn is doing right now ("executing tool: terminal"), keyed
// by STORED session id, from the `session.active_list` snapshot. The session
// list carries the same durable stamp, but the list is not re-read on an
// activity heartbeat (no `sessions.changed` fires for one), so a row reading
// only the list would freeze on its turn's first label. Each profile's poll
// owns its own bucket, so one profile's snapshot never clears another's rows.

let merged: Readonly<Record<string, string>> = {}
const byProfile = new Map<string, Record<string, string>>()

export const $liveActivityBySession = atom<Readonly<Record<string, string>>>({})

export function setLiveActivity(profileKey: string, activity: Record<string, string>): void {
  byProfile.set(profileKey, activity)
  merged = stableRecord(merged, Object.assign({}, ...byProfile.values()) as Record<string, string>)
  $liveActivityBySession.set(merged)
}
