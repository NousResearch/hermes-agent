import { ambientOwnerConnectionId, getApiRequestConnection } from '@/api/client'
import { getSession } from '@/hermes'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import type { SessionOwnerRoute, SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

export type StoredSessionProbe = { status: 'found'; session: SessionInfo } | { status: 'gone' | 'inconclusive' }

function isExplicitSessionMiss(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return /\b404\b/.test(message) && /session not found/i.test(message)
}

async function probeOwnedSession(
  storedSessionId: string,
  owner: SessionOwnerRoute,
  cached: SessionInfo | undefined
): Promise<StoredSessionProbe> {
  const cachedOwnerMatches =
    cached &&
    cached.connection_id === owner.connectionId &&
    (!cached.profile || normalizeProfileKey(cached.profile) === normalizeProfileKey(owner.profile))

  if (cached && cachedOwnerMatches) {
    return { status: 'found', session: cached }
  }

  try {
    const session = await getSession(storedSessionId, {
      connectionId: owner.connectionId,
      profile: owner.targetProfile || owner.profile
    })

    session.profile = normalizeProfileKey(owner.profile)
    session.connection_id = owner.connectionId

    return { status: 'found', session }
  } catch (error) {
    // An exact owner is fail-closed, including a persisted hint. Never turn a
    // failed read into a sweep of unrelated connections or profiles.
    return { status: isExplicitSessionMiss(error) ? 'gone' : 'inconclusive' }
  }
}

async function probeUnknownOwner(
  storedSessionId: string,
  cached: SessionInfo | undefined,
  knownOwner: SessionOwnerScope
): Promise<StoredSessionProbe> {
  // Inventory belongs to the source visible NOW. Registry probes retain both
  // halves of that route even if the window switches while a read is pending.
  const inventory = $profiles.get()
  const profiles = inventory.map(profile => normalizeProfileKey(profile.name))
  const connectionId = getApiRequestConnection()
  const ambientConnectionId = ambientOwnerConnectionId()
  const activeKey = normalizeProfileKey($activeGatewayProfile.get())

  // A legacy primary cannot be pinned without bypassing per-profile remote
  // overrides. Abandon its result if its source or owner changed instead.
  const legacyScopeChanged = () =>
    !connectionId &&
    (getApiRequestConnection() !== connectionId ||
      ambientOwnerConnectionId() !== ambientConnectionId ||
      $profiles.get() !== inventory ||
      knownOwnerForSession(storedSessionId) !== knownOwner)

  // A profile-less Recents row cannot route a multi-profile resume. Fall
  // through to by-id probes, which stamp ownership (#67603).
  if (cached && (cached.profile?.trim() || profiles.length <= 1)) {
    return { status: 'found', session: cached }
  }

  const candidates = [activeKey, ...profiles.filter(profile => profile !== activeKey)]
  let allGone = true

  for (const profile of candidates) {
    try {
      const scope = connectionId ? { connectionId, profile } : profile
      const session = await getSession(storedSessionId, scope)

      if (legacyScopeChanged()) {
        return { status: 'inconclusive' }
      }

      // Older backends can omit `profile` on the active lookup; preserve a
      // present stamp for compatibility. On another profile the DESKTOP alias
      // is authoritative: a per-profile remote override strips it before
      // forwarding, so that backend may answer as its own "default".
      if (profile === activeKey) {
        session.profile ||= activeKey
      } else {
        session.profile = profile
      }

      if (connectionId) {
        session.connection_id = connectionId
      }

      return { status: 'found', session }
    } catch (error) {
      allGone &&= isExplicitSessionMiss(error)

      if (legacyScopeChanged()) {
        return { status: 'inconclusive' }
      }
    }
  }

  // An inventory not yet loaded cannot vouch for ids on unlisted profiles.
  return { status: allGone && profiles.length > 0 ? 'gone' : 'inconclusive' }
}

export function probeSessionRoute(
  storedSessionId: string,
  cached: SessionInfo | undefined,
  ownerRoute?: SessionOwnerRoute
): Promise<StoredSessionProbe> {
  // Explicit caller tuples outrank hints; a bare profile is not an exact owner.
  const owner = ownerRoute ?? knownOwnerForSession(storedSessionId)

  return owner && typeof owner === 'object'
    ? probeOwnedSession(storedSessionId, owner, cached)
    : probeUnknownOwner(storedSessionId, cached, owner)
}
