export function normalizeProfileKey(name: null | string | undefined): string {
  const value = (name ?? '').trim()

  return value || 'default'
}

interface SessionIdentityLike {
  _lineage_root_id?: null | string
  id?: null | string
  profile?: null | string
}

const SESSION_IDENTITY_SEPARATOR = '\u0000'

/** Canonical identity for durable session state shared across profile backends. */
export function sessionIdentityKey(storedSessionId: string, profile: null | string | undefined): string {
  return `${normalizeProfileKey(profile)}${SESSION_IDENTITY_SEPARATOR}${storedSessionId.trim()}`
}

export function parseSessionIdentityKey(key: string): { profile: string; storedSessionId: string } {
  const separator = key.indexOf(SESSION_IDENTITY_SEPARATOR)

  if (separator < 0) {
    return { profile: 'default', storedSessionId: key }
  }

  return {
    profile: key.slice(0, separator),
    storedSessionId: key.slice(separator + SESSION_IDENTITY_SEPARATOR.length)
  }
}

/** Match a durable id (direct or lineage-root) only within its owning profile. */
export function sessionMatchesIdentity(
  session: SessionIdentityLike,
  storedSessionId: string,
  profile: null | string | undefined
): boolean {
  const targetId = storedSessionId.trim()

  if (!targetId || normalizeProfileKey(session.profile) !== normalizeProfileKey(profile)) {
    return false
  }

  return session.id === targetId || session._lineage_root_id === targetId
}
