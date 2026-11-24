import { randomBytes, timingSafeEqual } from 'node:crypto'

// One-time grants behind host.setProfileBackend's `prepare` and
// host.clearProfileBackend's `beforeApply`. The user confirms natively first;
// the plugin then starts (or stops) its backend; then the app applies the route
// change with the grant instead of asking again. A Cancel therefore starts or
// stops nothing, and the app never routes a profile to a backend that is not
// up yet, nor away from one the plugin has not stopped yet.
//
// A grant stands for the one dialog that minted it: it names the action, the
// plugin and the profile, is single use and expires. Renderer code can get one
// only through the native confirm, so it widens nothing.

export const PROFILE_BACKEND_GRANT_TTL_MS = 5 * 60_000

export type ProfileBackendGrantAction = 'clear' | 'set'

interface Grant {
  action: ProfileBackendGrantAction
  expires: number
  plugin: string
  profile: string
}

export interface ProfileBackendGrants {
  mint(action: ProfileBackendGrantAction, plugin: string, profile: string): string
  redeem(grant: unknown, action: ProfileBackendGrantAction, plugin: string, profile: string): boolean
  size(): number
}

export function createProfileBackendGrants(
  now: () => number = Date.now,
  ttlMs: number = PROFILE_BACKEND_GRANT_TTL_MS
): ProfileBackendGrants {
  const grants = new Map<string, Grant>()

  const expire = () => {
    const at = now()

    for (const [key, grant] of grants) {
      if (grant.expires <= at) {
        grants.delete(key)
      }
    }
  }

  return {
    mint(action, plugin, profile) {
      expire()
      const key = randomBytes(32).toString('base64url')
      grants.set(key, { action, expires: now() + ttlMs, plugin, profile })

      return key
    },

    redeem(presented, action, plugin, profile) {
      expire()

      if (typeof presented !== 'string' || !presented) {
        return false
      }

      const offered = Buffer.from(presented)

      for (const [key, grant] of grants) {
        const stored = Buffer.from(key)

        if (stored.length === offered.length && timingSafeEqual(stored, offered)) {
          // Spent on first presentation, even by the wrong caller.
          grants.delete(key)

          return grant.action === action && grant.plugin === plugin && grant.profile === profile
        }
      }

      return false
    },

    size() {
      expire()

      return grants.size
    }
  }
}
