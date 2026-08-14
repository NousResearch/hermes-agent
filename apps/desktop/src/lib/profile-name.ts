// Mirrors hermes_cli.profiles.validate_profile_name at renderer ingress points.
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const RESERVED_PROFILE_NAMES = new Set(['hermes', 'test', 'tmp', 'root', 'sudo'])

export function normalizeDesktopProfile(value: unknown): null | string {
  const profile = typeof value === 'string' ? value.trim().toLowerCase() : ''

  return profile && (profile === 'default' || (PROFILE_NAME_RE.test(profile) && !RESERVED_PROFILE_NAMES.has(profile)))
    ? profile
    : null
}
