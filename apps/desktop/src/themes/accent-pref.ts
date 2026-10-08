/**
 * Per-profile accent swatch id: `{ [profileKey]: id }`. No global slot: a profile without its own pick
 * paints its theme's accent, and a Bot Mode hop onto another profile must not inherit this one's color.
 * The record is profile-keyed, so it follows rename and delete with the profile's other state.
 */

import { persistStringRecord, storedStringRecord } from '@/lib/storage'

import { normalizeAccentId } from './accents'

export const PROFILE_ACCENTS_KEY = 'hermes-desktop-profile-accents-v1'

function without(profile: string): Record<string, string> {
  const { [profile]: _previous, ...others } = storedStringRecord(PROFILE_ACCENTS_KEY)

  return others
}

export const accentPref = {
  stored: (profile: string): null | string => normalizeAccentId(storedStringRecord(PROFILE_ACCENTS_KEY)[profile] ?? null),
  assign: (profile: string, value: null | string): void => {
    const id = normalizeAccentId(value)

    persistStringRecord(PROFILE_ACCENTS_KEY, id === null ? without(profile) : { ...without(profile), [profile]: id })
  },
  /** A renamed profile keeps its pick under the new name. */
  migrate: (from: string, to: string): void => {
    const id = accentPref.stored(from)

    accentPref.assign(from, null)
    accentPref.assign(to, id)
  },
  /** A deleted profile's pick goes with it, so a later profile of the same name starts on the theme accent. */
  drop: (profile: string): void => accentPref.assign(profile, null)
}
