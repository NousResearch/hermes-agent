import { describe, expect, it } from 'vitest'

import { petStartupCommand } from './tray-pet-policy'

describe('tray pet startup policy', () => {
  it('requests one pop-out only for an available inactive pet', () => {
    expect(petStartupCommand({ enabled: true, available: true, poppedOut: false, alreadyRequested: false })).toBe('pop-out')
    expect(petStartupCommand({ enabled: false, available: true, poppedOut: false, alreadyRequested: false })).toBeNull()
    expect(petStartupCommand({ enabled: true, available: false, poppedOut: false, alreadyRequested: false })).toBeNull()
    expect(petStartupCommand({ enabled: true, available: true, poppedOut: true, alreadyRequested: false })).toBeNull()
    expect(petStartupCommand({ enabled: true, available: true, poppedOut: false, alreadyRequested: true })).toBeNull()
  })
})
