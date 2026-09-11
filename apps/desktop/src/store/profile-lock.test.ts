import { webcrypto } from 'node:crypto'

import { beforeEach, describe, expect, it } from 'vitest'

import {
  clearProfilePasscode,
  hasProfilePasscode,
  isProfileLocked,
  noteProfileChange,
  PROFILE_LOCK_STORAGE_KEY,
  $profileLocks,
  setProfilePasscode,
  tryUnlockProfile,
  $unlockedProfile
} from './profile-lock'

if (!globalThis.crypto?.subtle) {
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: webcrypto, writable: true })
}

describe('profile lock store', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $profileLocks.set({})
    $unlockedProfile.set(null)
  })

  it('starts unlocked for profiles without a record', () => {
    expect(hasProfilePasscode('default')).toBe(false)
    expect(isProfileLocked('default')).toBe(false)
  })

  it('locks only the profile a passcode was set for', async () => {
    await setProfilePasscode('work', '1234')
    expect(hasProfilePasscode('work')).toBe(true)
    expect(isProfileLocked('work')).toBe(true)
    expect(isProfileLocked('default')).toBe(false)
  })

  it('persists the hashed record, never the plaintext passcode', async () => {
    await setProfilePasscode('work', 'hunter2')
    const raw = window.localStorage.getItem(PROFILE_LOCK_STORAGE_KEY)
    expect(raw).toBeTruthy()
    expect(raw).not.toContain('hunter2')
    expect(JSON.parse(raw!).work.algo).toBe('pbkdf2-sha256')
  })

  it('unlocks only with the correct passcode', async () => {
    await setProfilePasscode('work', '1234')
    expect(await tryUnlockProfile('work', 'nope')).toBe(false)
    expect(isProfileLocked('work')).toBe(true)
    expect(await tryUnlockProfile('work', '1234')).toBe(true)
    expect(isProfileLocked('work')).toBe(false)
  })

  it('forgets the unlock when the active profile changes away (re-prompt on return)', async () => {
    await setProfilePasscode('work', '1234')
    await tryUnlockProfile('work', '1234')
    expect(isProfileLocked('work')).toBe(false)
    noteProfileChange('default')
    expect(isProfileLocked('work')).toBe(true)
  })

  it('keeps the unlock when the note matches the unlocked profile', async () => {
    await setProfilePasscode('work', '1234')
    await tryUnlockProfile('work', '1234')
    noteProfileChange('work')
    expect(isProfileLocked('work')).toBe(false)
  })

  it('clears the lock and the persisted record', async () => {
    await setProfilePasscode('work', '1234')
    clearProfilePasscode('work')
    expect(hasProfilePasscode('work')).toBe(false)
    expect(isProfileLocked('work')).toBe(false)
    expect(window.localStorage.getItem(PROFILE_LOCK_STORAGE_KEY)).not.toContain('work')
  })
})
