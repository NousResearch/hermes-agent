// Guardrail for the ONE free-text door that creates a profile (the explicit
// "New profile" affordance behind the switcher's picker).
//
// A mistyped profile name is not a harmless empty directory: the first switch
// scaffolds a profile with a stock SOUL.md and boots a real backend for it
// (own serve port, sessions, cron state), so it presents as an agent nobody
// provisioned. The reported fleet drift was exactly this — `veste-frontend-dev`
// and `vestg-curriculum-admin` minted beside `vestr-frontend-dev` /
// `vestr-curriculum-admin`, one keystroke off the real names.
//
// Picking an existing profile is the picker's job; this is the backstop that
// notices when a typed name is a near-miss of one and says so before the
// scaffold happens.

export interface ProfileNameConflict {
  // `duplicate` — the name is taken (creation must not proceed).
  // `near-miss` — the name is one edit from an existing profile (warn + offer it).
  kind: 'duplicate' | 'near-miss'
  name: string
}

// Names shorter than this make one-edit-apart a meaningless signal ("res" is
// one edit from half the alphabet), so the near-miss warning stays off below it.
// Real profile slugs are far longer; the incident names were 16+ chars.
const MIN_NEAR_MISS_LENGTH = 4

/**
 * True when `a` and `b` differ by at most one edit: a substitution, a single
 * inserted/removed character, or one adjacent transposition ("defualt" →
 * "default"). Case-sensitive on purpose — callers compare normalized names.
 */
export function isOneEditApart(a: string, b: string): boolean {
  const short = a.length <= b.length ? a : b
  const long = a.length <= b.length ? b : a

  if (long.length - short.length > 1) {
    return false
  }

  if (short.length === long.length) {
    let first = -1

    for (let i = 0; i < short.length; i += 1) {
      if (short[i] !== long[i]) {
        first = i

        break
      }
    }

    if (first < 0) {
      return true // identical
    }

    for (let i = first + 1; i < short.length; i += 1) {
      if (short[i] === long[i]) {
        continue
      }

      // A second difference is only one edit away when it and the first form an
      // adjacent transposition; everything after must match again.
      if (i !== first + 1 || short[first] !== long[i] || short[i] !== long[first]) {
        return false
      }

      for (let j = i + 1; j < short.length; j += 1) {
        if (short[j] !== long[j]) {
          return false
        }
      }

      return true
    }

    return true
  }

  // One character longer: skip exactly one character in `long` and the rest must
  // line up.
  let i = 0

  while (i < short.length && short[i] === long[i]) {
    i += 1
  }

  for (let j = i; j < short.length; j += 1) {
    if (short[j] !== long[j + 1]) {
      return false
    }
  }

  return true
}

/**
 * Classify a typed profile name against the profiles that already exist.
 *
 * Returns the taken name for an exact (case-insensitive) match, else the
 * alphabetically-first near-miss so the dialog can nudge to it deterministically
 * (the ordering never depends on the caller's profile-list order). Null means
 * the name is safe to create.
 */
export function findProfileNameConflict(candidate: string, existing: readonly string[]): null | ProfileNameConflict {
  const typed = candidate.trim().toLowerCase()

  if (!typed) {
    return null
  }

  const names = [...new Set(existing.map(name => name.trim().toLowerCase()).filter(Boolean))].sort()

  if (names.includes(typed)) {
    return { kind: 'duplicate', name: typed }
  }

  const near = names.find(
    name => typed.length >= MIN_NEAR_MISS_LENGTH && name.length >= MIN_NEAR_MISS_LENGTH && isOneEditApart(typed, name)
  )

  return near ? { kind: 'near-miss', name: near } : null
}
