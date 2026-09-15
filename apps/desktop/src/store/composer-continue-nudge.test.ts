import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.composer.continueOnDoubleEnter'

const loadStore = () => import('./composer-continue-nudge')

describe('continue-on-double-Enter preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('defaults off, so an empty Enter never sends anything until the user opts in', async () => {
    const store = await loadStore()

    expect(store.$continueOnDoubleEnter.get()).toBe(false)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('persists the opt-in across a reload', async () => {
    const store = await loadStore()

    store.setContinueOnDoubleEnter(true)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('true')

    vi.resetModules()
    const reloaded = await loadStore()

    expect(reloaded.$continueOnDoubleEnter.get()).toBe(true)
  })
})
