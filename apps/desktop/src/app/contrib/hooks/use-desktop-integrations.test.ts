import { describe, expect, it } from 'vitest'

import { shouldPersistRememberedLocation, shouldRestoreRememberedLocation } from './use-desktop-integrations'

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

describe('shouldPersistRememberedLocation', () => {
  it('does not let a new-session window overwrite shared remembered history', () => {
    expect(shouldPersistRememberedLocation(true)).toBe(false)
  })

  it('keeps the primary window updating remembered history', () => {
    expect(shouldPersistRememberedLocation(false)).toBe(true)
  })
})
