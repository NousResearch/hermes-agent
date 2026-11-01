import { afterEach, describe, expect, it, vi } from 'vitest'

import { $focusFollowsMouse, setFocusFollowsMouse } from './focus-follows-mouse'

const KEY = 'hermes.desktop.focusFollowsMouse'

afterEach(() => {
  setFocusFollowsMouse(false)
  window.localStorage.removeItem(KEY)
})

describe('focus follows mouse preference', () => {
  it('defaults to explicit selection without writing a provisional default', () => {
    expect($focusFollowsMouse.get()).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBeNull()
  })

  it('persists and reloads either choice, and follows newer storage values live', async () => {
    setFocusFollowsMouse(true)
    expect(window.localStorage.getItem(KEY)).toBe('true')
    vi.resetModules()
    const reloaded = await import('./focus-follows-mouse')
    expect(reloaded.$focusFollowsMouse.get()).toBe(true)
    reloaded.setFocusFollowsMouse(false)
    expect(window.localStorage.getItem(KEY)).toBe('false')
    vi.resetModules()
    expect((await import('./focus-follows-mouse')).$focusFollowsMouse.get()).toBe(false)
    window.localStorage.setItem(KEY, 'true')
    // An older queued event must read the actual current stored value.
    window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: 'false' }))
    expect($focusFollowsMouse.get()).toBe(true)
    window.localStorage.removeItem(KEY)
    window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: null }))
    expect($focusFollowsMouse.get()).toBe(false)
  })
})
