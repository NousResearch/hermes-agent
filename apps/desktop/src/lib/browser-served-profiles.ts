import type { HermesApiRequest } from '@/global'
import { browserLaunchProfileOrDefault } from '@/lib/platform'

// The multiplexer applies a park, unpark or standalone change on its 30 s profile rescan, and none
// of those changes the installed list, so a known served set is re-read on that cadence.
const SERVED_SET_TTL_MS = 30_000
// An inventory read waits at most this long for the set, then answers with the set it already has.
// A pick's refresh of the set is capped at the same 5 s, well under the store's 20 s budget for a
// dial's URL, so a profile that stays unserved is still refused and parked rather than timed out.
export const SERVED_SET_READ_TIMEOUT_MS = 5_000

const profileKey = (profile?: null | string) => profile?.trim() || 'default'

/**
 * The tab refused to open a socket for a profile its gateway does not serve. No redial can change
 * the answer, since serving the profile takes a change on the gateway, so the store parks the dial
 * on it until the next explicit open asks again.
 */
export class UnservedProfileError extends Error {
  constructor(readonly profile: string) {
    super(`This gateway does not serve profile "${profile}", so this tab cannot open it.`)
    this.name = 'UnservedProfileError'
  }
}

/**
 * The profiles the gateway serving this tab answers for. `/api/profiles` lists every installed
 * profile, but the tab's one socket reaches only the gateway's own profile and, on a multiplexer,
 * the rest of its served set (`/api/status` `gateway_shared_with`). A standalone or parked profile,
 * and every other profile of a single-profile gateway, is installed here but not served here.
 *
 * Only an inventory read (`refresh`) fetches the set. Routing reads the cached answer (`serves`)
 * and never waits on the network. Until a gateway reports the set, every installed profile counts
 * as served, which is how the tab treated them before the set was read.
 */
export function createServedProfiles(api: <T>(request: HermesApiRequest) => Promise<T>) {
  let served: null | Set<string> = null
  // The installed list the last read went with. A reconnect clears it so the next read re-reads.
  let readFor: null | string = null
  let readAt = 0
  let pending: null | Promise<void> = null

  const read = async () => {
    try {
      const status = await api<{ gateway_shared_with?: null | string[] }>({
        path: '/api/status',
        timeoutMs: SERVED_SET_READ_TIMEOUT_MS
      })

      const shared = status?.gateway_shared_with

      // Absent on a backend older than the field; [] on a gateway that serves only its own profile.
      served = Array.isArray(shared)
        ? new Set([...shared.map(name => profileKey(String(name))), browserLaunchProfileOrDefault()])
        : null
    } catch {
      // A failed read keeps the last answer: a status blip must not change which profiles are reachable.
    }
  }

  return {
    invalidate: () => {
      readFor = null
    },
    /** Re-read the set when the installed list changed since the last read, or the read is stale. */
    refresh: async (installed: string[]): Promise<void> => {
      const key = installed.map(profileKey).sort().join('\n')

      // A read in flight may have gone out for an older list or before a reconnect, so wait for it,
      // then check again: its answer settles this list only if it was read for this list.
      while (pending) {
        await pending
      }

      if (key === readFor && Date.now() - readAt < SERVED_SET_TTL_MS) {
        return
      }

      readFor = key
      pending = read().finally(() => {
        readAt = Date.now()
        pending = null
      })

      await pending
    },
    serves: (profile?: null | string) => served === null || served.has(profileKey(profile))
  }
}
