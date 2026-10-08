import { beforeEach, describe, expect, it } from 'vitest'

import { dropTilesForProfile, migrateTilesForProfile } from '@/store/session-states'
import { accentPref } from '@/themes/accent-pref'

beforeEach(() => window.localStorage.clear())

describe('profile accent follows the profile', () => {
  it('moves with a rename', () => {
    accentPref.assign('work', 'pink')

    migrateTilesForProfile('work', 'studio')

    expect(accentPref.stored('studio')).toBe('pink')
    expect(accentPref.stored('work')).toBeNull()
  })

  it('is dropped with the profile, so a new profile of that name starts on the theme accent', () => {
    accentPref.assign('work', 'pink')
    accentPref.assign('default', 'green')

    dropTilesForProfile('work')

    expect(accentPref.stored('work')).toBeNull()
    expect(accentPref.stored('default')).toBe('green')
  })
})
