import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $profileSwitchBehavior,
  $profileSwitchRestoreIntent,
  _resetProfileSwitchBehaviorForTests,
  clearProfileSwitchRestoreIntent,
  getProfileSwitchBehavior,
  registerProfileSwitchAnchorCapture,
  requestProfileSwitchRestore,
  scopeProfileSwitchRestoreIntent,
  setProfileSwitchBehavior
} from './profile-switch-behavior'

describe('profile switch behavior preference', () => {
  beforeEach(() => {
    localStorage.clear()
    _resetProfileSwitchBehaviorForTests()
  })

  afterEach(() => {
    localStorage.clear()
    _resetProfileSwitchBehaviorForTests()
  })

  it('defaults to fresh_draft when no preference exists', () => {
    expect(getProfileSwitchBehavior()).toBe('fresh_draft')
    expect($profileSwitchBehavior.get()).toBe('fresh_draft')
  })

  it('persists restore_last_session as a client-local preference', () => {
    setProfileSwitchBehavior('restore_last_session')

    expect($profileSwitchBehavior.get()).toBe('restore_last_session')
    expect(localStorage.getItem('hermes.desktop.profile-switch-behavior.v1')).toBe('restore_last_session')
  })

  it('captures the visible anchor and replaces an older intent with the latest profile', () => {
    let captures = 0
    registerProfileSwitchAnchorCapture(() => {
      captures += 1
    })
    requestProfileSwitchRestore('beta')
    const first = $profileSwitchRestoreIntent.get()
    requestProfileSwitchRestore('gamma')
    const latest = $profileSwitchRestoreIntent.get()

    expect(first).not.toBeNull()
    expect(captures).toBe(2)
    expect(latest).toMatchObject({ profile: 'gamma' })
    expect(latest?.sequence).toBeGreaterThan(first?.sequence ?? 0)

    scopeProfileSwitchRestoreIntent(first?.sequence ?? 0, 'remote-a')
    expect($profileSwitchRestoreIntent.get()?.connectionId).toBeUndefined()

    scopeProfileSwitchRestoreIntent(latest?.sequence ?? 0, 'remote-b')
    expect($profileSwitchRestoreIntent.get()?.connectionId).toBe('remote-b')
  })

  it('clears only the matching latest restore intent', () => {
    registerProfileSwitchAnchorCapture(() => undefined)
    requestProfileSwitchRestore('beta')
    const first = $profileSwitchRestoreIntent.get()
    requestProfileSwitchRestore('gamma')

    clearProfileSwitchRestoreIntent(first?.sequence ?? 0)
    expect($profileSwitchRestoreIntent.get()?.profile).toBe('gamma')

    clearProfileSwitchRestoreIntent($profileSwitchRestoreIntent.get()?.sequence ?? 0)
    expect($profileSwitchRestoreIntent.get()).toBeNull()
  })
})
