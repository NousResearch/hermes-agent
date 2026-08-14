import { describe, expect, it } from 'vitest'

import { isValidProfileName } from './create-profile-dialog'

describe('isValidProfileName', () => {
  it('accepts canonical named profiles and the default alias', () => {
    expect(isValidProfileName('life_2')).toBe(true)
    expect(isValidProfileName(' default ')).toBe(true)
  })

  it('rejects malformed and non-default CLI reserved names', () => {
    for (const profile of ['../life', 'bad profile', 'hermes', 'test', 'tmp', 'root', 'sudo']) {
      expect(isValidProfileName(profile), profile).toBe(false)
    }
  })
})
