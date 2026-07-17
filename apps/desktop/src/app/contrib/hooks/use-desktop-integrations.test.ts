import { describe, expect, it } from 'vitest'

import { shouldRestoreRememberedLocation } from './use-desktop-integrations'

describe('shouldRestoreRememberedLocation', () => {
  it('does not restore shared renderer history into a new-session window', () => {
    expect(shouldRestoreRememberedLocation('/', true)).toBe(false)
  })

  it('still restores the primary window when it boots on the new-chat route', () => {
    expect(shouldRestoreRememberedLocation('/', false)).toBe(true)
  })

  it('never replaces an explicit routed session pop-out', () => {
    expect(shouldRestoreRememberedLocation('/stored-session', false)).toBe(false)
  })
})
