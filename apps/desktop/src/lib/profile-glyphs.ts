// Optional per-profile glyph override (the rail squares and the home pill).
// Codicons are role labels a user picks at creation time (#79233); absent
// overrides keep the built-in marks — `home` for the default profile, the
// initial for named ones.

// Curated subset of the codicon catalog covering the roles profiles actually
// play (work, staging, research, ops…). The full catalog is overwhelming in a
// picker; this vocabulary stays scannable and matches the common names users
// give their profiles.
export const PROFILE_GLYPHS: readonly string[] = [
  'home',
  'briefcase',
  'rocket',
  'beaker',
  'terminal',
  'shield',
  'lock',
  'key',
  'database',
  'server',
  'cloud',
  'graph',
  'book',
  'paintcan',
  'lightbulb',
  'bug',
  'tools',
  'gear',
  'circuit-board',
  'plug',
  'organization',
  'account'
]

// A profile's effective glyph: its stored override, else null (callers fall
// back to their built-in mark). Keys are trimmed to match how every other
// per-profile preference addresses a profile.
export function resolveProfileGlyph(name: null | string | undefined, overrides: Record<string, string>): string | null {
  const key = (name ?? '').trim()

  if (!key) {
    return null
  }

  return overrides[key] ?? null
}
