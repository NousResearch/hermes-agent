// Mirrors hermes_cli.profiles._PROFILE_ID_RE so the desktop never routes or
// spawns a backend with a profile name the Python resolver would reject.
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

// Keep this in lockstep with hermes_cli.profiles._RESERVED_NAMES. `default`
// remains the one valid built-in alias; the other names are refused before a
// route or child-process argv can carry them into the CLI.
const RESERVED_PROFILE_NAMES = new Set(['hermes', 'test', 'tmp', 'root', 'sudo'])

/** Return a canonical desktop profile name, or null for absent/malformed input. */
export function normalizeDesktopProfile(value: unknown): null | string {
  const profile = typeof value === 'string' ? value.trim().toLowerCase() : ''

  return profile && (profile === 'default' || (PROFILE_NAME_RE.test(profile) && !RESERVED_PROFILE_NAMES.has(profile)))
    ? profile
    : null
}
