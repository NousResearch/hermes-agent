import { beforeEach, describe, expect, it } from 'vitest'

import { $vibeHeartsEnabled, setVibeHeartsEnabled } from '@/store/vibe-hearts-enabled'

const KEY = 'hermes.desktop.vibeHearts.v1'

describe('vibe hearts enabled', () => {
  beforeEach(() => {
    window.localStorage.removeItem(KEY)
    // Reset to the pre-toggle always-on default without going through the
    // setter's persist path first (absent key means on).
    $vibeHeartsEnabled.set(true)
  })

  it('defaults to on so existing installs keep hearts', () => {
    expect($vibeHeartsEnabled.get()).toBe(true)
  })

  it('turns off and persists', () => {
    setVibeHeartsEnabled(false)
    expect($vibeHeartsEnabled.get()).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBe('off')
  })

  it('turns back on and persists', () => {
    setVibeHeartsEnabled(false)
    setVibeHeartsEnabled(true)
    expect($vibeHeartsEnabled.get()).toBe(true)
    expect(window.localStorage.getItem(KEY)).toBe('on')
  })
})
