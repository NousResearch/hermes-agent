import { normalizeProfileKey } from './profile-key'

export interface SessionIdentity {
  profile: string
  sessionId: string
}

export function makeSessionIdentity(
  profile: string | null | undefined,
  sessionId: string
): SessionIdentity {
  return { profile: normalizeProfileKey(profile), sessionId }
}

export function sessionIdentityEqual(left: SessionIdentity, right: SessionIdentity): boolean {
  return normalizeProfileKey(left.profile) === normalizeProfileKey(right.profile) && left.sessionId === right.sessionId
}

export function sessionIdentityKey(identity: SessionIdentity): string {
  return JSON.stringify([normalizeProfileKey(identity.profile), identity.sessionId])
}

export function profileSessionKey(profile: string | null | undefined, sessionId: string): string {
  return sessionIdentityKey(makeSessionIdentity(profile, sessionId))
}

export function parseSessionIdentityKey(value: string): SessionIdentity | null {
  try {
    const parsed: unknown = JSON.parse(value)

    if (
      Array.isArray(parsed) &&
      parsed.length === 2 &&
      typeof parsed[0] === 'string' &&
      typeof parsed[1] === 'string'
    ) {
      return makeSessionIdentity(parsed[0], parsed[1])
    }
  } catch {
    // Malformed persisted/cache data is simply not an identity.
  }

  return null
}

export function hasSessionIdentity(
  identities: readonly SessionIdentity[],
  identity: SessionIdentity
): boolean {
  return identities.some(candidate => sessionIdentityEqual(candidate, identity))
}

export function toggleSessionIdentity(
  identities: SessionIdentity[],
  identity: SessionIdentity,
  on: boolean
): SessionIdentity[] {
  const present = hasSessionIdentity(identities, identity)

  if (on) {
    return present ? identities : [...identities, makeSessionIdentity(identity.profile, identity.sessionId)]
  }

  return present ? identities.filter(candidate => !sessionIdentityEqual(candidate, identity)) : identities
}

export function getProfileSessionValue<T>(
  map: ReadonlyMap<string, T>,
  profile: string | null | undefined,
  sessionId: string
): T | undefined {
  return map.get(profileSessionKey(profile, sessionId))
}

export function setProfileSessionValue<T>(
  map: Map<string, T>,
  profile: string | null | undefined,
  sessionId: string,
  value: T
): void {
  map.set(profileSessionKey(profile, sessionId), value)
}

export function deleteProfileSessionValue(
  map: Map<string, unknown>,
  profile: string | null | undefined,
  sessionId: string
): boolean {
  return map.delete(profileSessionKey(profile, sessionId))
}
