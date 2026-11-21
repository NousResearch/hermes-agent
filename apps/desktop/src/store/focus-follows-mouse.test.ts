import { afterEach, describe, expect, it, vi } from 'vitest'

import { $focusFollowsMouse, setFocusFollowsMouse } from './focus-follows-mouse'

const KEY = 'hermes.desktop.focusFollowsMouse'

afterEach(() => {
  setFocusFollowsMouse(true)
  window.localStorage.removeItem(KEY)
})

describe('focus follows mouse preference', () => {
  it('defaults to the documented pointer behaviour without writing a provisional default', () => {
    expect($focusFollowsMouse.get()).toBe(true)
    expect(window.localStorage.getItem(KEY)).toBeNull()
  })

  it('persists and reloads either choice, and follows newer storage values live', async () => {
    setFocusFollowsMouse(false)
    expect(window.localStorage.getItem(KEY)).toBe('false')
    vi.resetModules()
    const reloaded = await import('./focus-follows-mouse')
    expect(reloaded.$focusFollowsMouse.get()).toBe(false)
    reloaded.setFocusFollowsMouse(true)
    expect(window.localStorage.getItem(KEY)).toBe('true')
    vi.resetModules()
    expect((await import('./focus-follows-mouse')).$focusFollowsMouse.get()).toBe(true)
    window.localStorage.setItem(KEY, 'false')
    // An older queued event must read the actual current stored value.
    window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: 'true' }))
    expect($focusFollowsMouse.get()).toBe(false)
    window.localStorage.removeItem(KEY)
    window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: null }))
    // Removing the key falls back to the documented pointer behaviour.
    expect($focusFollowsMouse.get()).toBe(true)
  })
})
