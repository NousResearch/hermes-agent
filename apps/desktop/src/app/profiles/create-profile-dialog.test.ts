import { describe, expect, it } from 'vitest'

import { isValidProfileName } from './create-profile-dialog'

describe('isValidProfileName', () => {
  it('accepts lowercase names', () => {
    expect(isValidProfileName('my-profile')).toBe(true)
    expect(isValidProfileName('test123')).toBe(true)
    expect(isValidProfileName('a')).toBe(true)
  })

  it('rejects uppercase names', () => {
    expect(isValidProfileName('MyProfile')).toBe(false)
    expect(isValidProfileName('Test')).toBe(false)
  })

  it('rejects names starting with hyphen or underscore', () => {
    expect(isValidProfileName('-bad')).toBe(false)
    expect(isValidProfileName('_bad')).toBe(false)
  })

  it('trims whitespace before validation', () => {
    expect(isValidProfileName('  my-profile  ')).toBe(true)
  })

  it('rejects empty or whitespace-only names', () => {
    expect(isValidProfileName('')).toBe(false)
    expect(isValidProfileName('   ')).toBe(false)
  })
})
