import { afterEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.hideThreadTimeline'

afterEach(() => {
  window.localStorage.removeItem(STORAGE_KEY)
  vi.resetModules()
})

describe('thread timeline preference', () => {
  it('keeps bars visible by default and remembers both choices across reloads', async () => {
    window.localStorage.removeItem(STORAGE_KEY)
    vi.resetModules()
    const initial = await import('./thread-timeline')

    expect(initial.$hideThreadTimeline.get()).toBe(false)
    initial.setHideThreadTimeline(true)

    vi.resetModules()
    const hidden = await import('./thread-timeline')
    expect(hidden.$hideThreadTimeline.get()).toBe(true)
    hidden.setHideThreadTimeline(false)

    vi.resetModules()
    const visible = await import('./thread-timeline')
    expect(visible.$hideThreadTimeline.get()).toBe(false)
  })
})
