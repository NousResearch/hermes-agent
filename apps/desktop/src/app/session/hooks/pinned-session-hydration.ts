import type { SessionInfo } from '@/hermes'
import { mapPool } from '@/lib/pool'

const DEFAULT_PROFILE = 'default'
const ALL_PROFILES_SCOPE = '__all__'
export const MAX_PIN_HYDRATION_IDS = 100
export const PIN_HYDRATION_CONCURRENCY = 4

export type GetSessionByProfile = (id: string, profile: null | string) => Promise<SessionInfo>

export function missingPinnedSessionIds(pinIds: string[], loaded: SessionInfo[]): string[] {
  const loadedKeys = new Set(loaded.flatMap(session => [session.id, session._lineage_root_id].filter(Boolean)))

  return pinIds.filter(id => !loadedKeys.has(id))
}

export function pinHydrationProfiles(
  profileScope: string,
  knownProfiles: string[],
  allProfilesScope = ALL_PROFILES_SCOPE
): string[] {
  if (profileScope !== allProfilesScope) {
    return [profileScope.trim() || DEFAULT_PROFILE]
  }

  return [...new Set([DEFAULT_PROFILE, ...knownProfiles.map(name => name.trim()).filter(Boolean)])]
}

/** Resolve persisted pin ids into real rows without depending on the recent-page window. */
export async function hydratePinnedSessions(
  pinIds: string[],
  profiles: string[],
  getSession: GetSessionByProfile
): Promise<SessionInfo[]> {
  const boundedPinIds = [...new Set(pinIds)].slice(0, MAX_PIN_HYDRATION_IDS)

  const resolved = await mapPool(boundedPinIds, PIN_HYDRATION_CONCURRENCY, async id => {
    for (const profile of profiles) {
      try {
        const session = await getSession(id, profile)

        if (session.archived) {
          return null
        }

        // The explicitly probed Desktop profile owns the row. A per-profile
        // remote override answers as its own "default", which must not leak
        // into Desktop routing. Also treat a historical local pin as the
        // pending boot migration it is: `pinned:false` must not pull-delete
        // the local pin before watchSessionPins can PATCH it true.
        return {
          ...session,
          pinned: session.pinned === true ? true : undefined,
          profile
        } as SessionInfo
      } catch {
        // A pin can belong to another profile or point at a removed row.
      }
    }

    return null
  })

  return resolved.filter((session): session is SessionInfo => session !== null)
}
