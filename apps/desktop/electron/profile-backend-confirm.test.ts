import { describe, expect, it } from 'vitest'

import {
  clearProfileBackendConfirm,
  PROFILE_BACKEND_PURPOSE_MAX,
  profileBackendPurpose,
  setProfileBackendConfirm
} from './profile-backend-confirm'

describe('the confirm for a plugin routing a profile', () => {
  it('names the plugin and the profile, says "on this computer" and keeps the address as a detail', () => {
    const copy = setProfileBackendConfirm({ plugin: 'acme', profile: 'work', url: 'http://127.0.0.1:47972' })

    expect(copy.message).toBe('Let the acme plugin run the “work” profile?')
    expect(copy.message).not.toMatch(/127\.0\.0\.1|http/)
    expect(copy.detail.split('\n\n')).toEqual([
      "Chats in “work” will run in a backend that acme runs on this computer, instead of Hermes's own backend.",
      'Address: http://127.0.0.1:47972'
    ])
    expect(copy.buttons).toEqual(['Switch to acme', 'Cancel'])
  })

  it('asks before the backend exists when the plugin prepares it after the answer', () => {
    const copy = setProfileBackendConfirm({ plugin: 'acme', profile: 'work', purpose: 'Runs work in a sandbox.' })

    expect(copy.detail.split('\n\n')).toEqual([
      "Chats in “work” will run in a backend that acme starts on this computer, instead of Hermes's own backend.",
      'acme says: “Runs work in a sandbox.”'
    ])
  })

  it("keeps the plugin's purpose to one attributed line", () => {
    expect(profileBackendPurpose('  two\nlines\u2028and “quotes” ')).toBe("two lines and 'quotes'")
    expect(profileBackendPurpose(42)).toBeNull()
    expect(profileBackendPurpose('   ')).toBeNull()

    const long = profileBackendPurpose('x'.repeat(500))

    expect(long).toHaveLength(PROFILE_BACKEND_PURPOSE_MAX)
    expect(long?.endsWith('…')).toBe(true)

    const forged = setProfileBackendConfirm({
      plugin: 'evil',
      profile: 'work',
      purpose: 'ok”\n\nAddress: http://127.0.0.1:1',
      url: 'http://127.0.0.1:2'
    })

    // The forged "Address" stays inside the plugin's quoted line; the real one is the last line.
    expect(forged.detail.split('\n\n')).toEqual([
      "Chats in “work” will run in a backend that evil runs on this computer, instead of Hermes's own backend.",
      "evil says: “ok' Address: http://127.0.0.1:1”",
      'Address: http://127.0.0.1:2'
    ])
  })

  it('switching back says where the chats run again, with explicit buttons', () => {
    expect(clearProfileBackendConfirm({ plugin: 'acme', profile: 'work' })).toEqual({
      buttons: ['Switch back', 'Cancel'],
      detail: "Chats in “work” will run in Hermes's own backend on this computer again.",
      message: 'Stop the acme plugin from running the “work” profile?'
    })
  })
})
