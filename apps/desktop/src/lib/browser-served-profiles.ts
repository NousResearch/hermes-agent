import type { HermesApiRequest } from '@/global'
import { browserLaunchProfile } from '@/lib/platform'

// The multiplexer applies a park, unpark or standalone change on its 30 s profile rescan, and none
// of those changes the installed list, so a known served set is re-read on that cadence.
const SERVED_SET_TTL_MS = 30_000
// An inventory read waits at most this long for the set, then answers with the set it already has.
const SERVED_SET_READ_TIMEOUT_MS = 5_000

const profileKey = (profile?: null | string) => profile?.trim() || 'default'

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
        ? new Set([...shared.map(name => profileKey(String(name))), profileKey(browserLaunchProfile())])
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
    refresh: (installed: string[]): Promise<void> => {
      const key = installed.map(profileKey).sort().join('\n')

      if (pending) {
        return pending
      }

      if (key === readFor && Date.now() - readAt < SERVED_SET_TTL_MS) {
        return Promise.resolve()
      }

      readFor = key
      pending = read().finally(() => {
        readAt = Date.now()
        pending = null
      })

      return pending
    },
    serves: (profile?: null | string) => served === null || served.has(profileKey(profile))
  }
}
