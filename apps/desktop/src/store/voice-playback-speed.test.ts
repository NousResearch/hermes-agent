import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.voicePlaybackSpeed'

const loadStore = () => import('./voice-playback-speed')

describe('voice playback speed preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('defaults to 1x with no stored record', async () => {
    const store = await loadStore()

    expect(store.$voicePlaybackSpeed.get()).toBe(1)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('persists a user-chosen rate and seeds a fresh store from it', async () => {
    const store = await loadStore()

    store.setVoicePlaybackSpeed(1.5)

    expect(store.$voicePlaybackSpeed.get()).toBe(1.5)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('1.5')

    vi.resetModules()
    const fresh = await loadStore()

    expect(fresh.$voicePlaybackSpeed.get()).toBe(1.5)
  })

  it('falls back to 1x for out-of-range or malformed stored values', async () => {
    window.localStorage.setItem(STORAGE_KEY, '250')
    let store = await loadStore()

    expect(store.$voicePlaybackSpeed.get()).toBe(1)

    vi.resetModules()
    window.localStorage.setItem(STORAGE_KEY, 'fast')
    store = await loadStore()

    expect(store.$voicePlaybackSpeed.get()).toBe(1)

    vi.resetModules()
    window.localStorage.setItem(STORAGE_KEY, '0.1')
    store = await loadStore()

    expect(store.$voicePlaybackSpeed.get()).toBe(1)
  })

  it('ignores out-of-range writes instead of clamping them', async () => {
    const store = await loadStore()

    store.setVoicePlaybackSpeed(9)
    expect(store.$voicePlaybackSpeed.get()).toBe(1)

    store.setVoicePlaybackSpeed(0.1)
    expect(store.$voicePlaybackSpeed.get()).toBe(1)
    expect(store.isVoicePlaybackSpeed(4)).toBe(true)
  })

  // Returning to the default removes the record instead of storing "1" —
  // same contract as the video playback speed store.
  it('drops the stored key when the rate returns to the default', async () => {
    const store = await loadStore()

    store.setVoicePlaybackSpeed(2)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('2')

    store.setVoicePlaybackSpeed(1)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })
})
