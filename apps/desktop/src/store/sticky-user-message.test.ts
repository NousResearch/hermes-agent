import { afterEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.stickyUserMessage'

afterEach(() => {
  window.localStorage.removeItem(STORAGE_KEY)
  vi.resetModules()
})

describe('sticky user message preference', () => {
  it('pins by default and remembers both choices across reloads', async () => {
    window.localStorage.removeItem(STORAGE_KEY)
    vi.resetModules()
    const initial = await import('./sticky-user-message')

    expect(initial.$stickyUserMessage.get()).toBe(true)
    initial.setStickyUserMessage(false)

    vi.resetModules()
    const off = await import('./sticky-user-message')
    expect(off.$stickyUserMessage.get()).toBe(false)
    off.setStickyUserMessage(true)

    vi.resetModules()
    const pinned = await import('./sticky-user-message')
    expect(pinned.$stickyUserMessage.get()).toBe(true)
  })
})
