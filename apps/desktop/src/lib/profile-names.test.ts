import { describe, expect, it } from 'vitest'

import { isReservedProfileName, isValidProfileName, profileNameHint } from './profile-names'

const copy = {
  nameHint: 'syntax hint',
  reservedNameHint: 'reserved hint'
}

describe('profile name validation', () => {
  it('accepts valid profile names', () => {
    expect(isValidProfileName('coder')).toBe(true)
    expect(isValidProfileName('team-agent_2')).toBe(true)
  })

  it('rejects backend reserved names', () => {
    expect(isReservedProfileName('test')).toBe(true)
    expect(isReservedProfileName(' sudo ')).toBe(true)
    expect(isValidProfileName('test')).toBe(false)
  })

  it('uses reserved-name help only for syntactically valid reserved names', () => {
    expect(profileNameHint('test', copy)).toBe('reserved hint')
    expect(profileNameHint('Test', copy)).toBe('syntax hint')
  })
})
