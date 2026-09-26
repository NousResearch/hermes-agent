// Single source of truth for built-in personality names on the desktop.
// Mirrors hermes_cli/personality.py BUILTIN_PERSONALITIES — the backend
// single owner. Keep in sync when a built-in is added there.
export const BUILTIN_PERSONALITIES = [
  'helpful',
  'concise',
  'technical',
  'creative',
  'teacher',
  'kawaii',
  'catgirl',
  'pirate',
  'shakespeare',
  'surfer',
  'noir',
  'uwu',
  'philosopher',
  'hype'
]

// Spellings the runtime treats as "no personality" — mirrors
// hermes_cli/personality.py NEUTRAL_PERSONALITY_NAMES. A config key with any of
// these (after folding) never resolves to a personality, so it must not be
// offered in the dropdown.
export const NEUTRAL_PERSONALITY_NAMES = new Set(['', 'none', 'default', 'neutral'])

/**
 * Canonical personality key, mirroring hermes_cli/personality.py
 * `normalize_personality_name` (`str(name).strip().lower()`, neutral spellings →
 * ''). The runtime folds every user-config key this way before resolving it, so
 * any reader that lists names for selection must fold identically — otherwise the
 * dropdown offers rows the runtime can never resolve (a case-variant duplicate, a
 * whitespace-padded name, or a neutral spelling like `none`/`default`/`neutral`).
 * Returns '' for a neutral/blank name (caller skips it).
 */
export function foldPersonalityName(name: unknown): string {
  const key = String(name ?? '').trim().toLowerCase()
  return NEUTRAL_PERSONALITY_NAMES.has(key) ? '' : key
}
