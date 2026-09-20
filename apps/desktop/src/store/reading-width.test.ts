import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.readingWidth'

describe('reading width preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('persists a selected reading width and restores it on reload', async () => {
    const first = await import('./reading-width')

    first.setReadingWidth('comfortable')

    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('comfortable')
    first.setReadingWidth('wide')
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('wide')
    expect(first.$readingWidth.get()).toBe('wide')
  })

  it('falls back to wide for an unknown stored value', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'extra-wide')
    vi.resetModules()

    const store = await import('./reading-width')

    expect(store.$readingWidth.get()).toBe('wide')
  })
})
