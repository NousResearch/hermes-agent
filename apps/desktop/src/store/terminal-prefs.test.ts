import { beforeEach, describe, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.revealBackgroundTerminals'

describe('revealBackgroundTerminals preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('defaults to stack when the key is missing', async () => {
    const { $revealBackgroundTerminals } = await import('./terminal-prefs')

    expect($revealBackgroundTerminals.get()).toBe('stack')
  })

  it('treats invalid stored values as stack', async () => {
    window.localStorage.setItem(KEY, 'always')
    const { $revealBackgroundTerminals } = await import('./terminal-prefs')

    expect($revealBackgroundTerminals.get()).toBe('stack')
  })

  it('persists auto and restores it on reload', async () => {
    const { setRevealBackgroundTerminals } = await import('./terminal-prefs')
    setRevealBackgroundTerminals('auto')
    expect(window.localStorage.getItem(KEY)).toBe('auto')

    vi.resetModules()
    const { $revealBackgroundTerminals } = await import('./terminal-prefs')
    expect($revealBackgroundTerminals.get()).toBe('auto')
  })
})
