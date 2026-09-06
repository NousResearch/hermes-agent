import { beforeEach, describe, expect, it, vi } from 'vitest'

const LEGACY_VISIBLE_KEY = 'hermes.desktop.statusbarVisible'

const loadStore = () => import('./statusbar-prefs')

describe('statusbar whole-bar visibility', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('shows the bar on a fresh install and for installs that hid it under the v1 key', async () => {
    window.localStorage.setItem(LEGACY_VISIBLE_KEY, 'false')

    const { $statusbarVisible } = await loadStore()

    expect($statusbarVisible.get()).toBe(true)
  })

  it('still honours a hide made after the update', async () => {
    const first = await loadStore()

    first.toggleStatusbarVisible()
    expect(first.$statusbarVisible.get()).toBe(false)

    vi.resetModules()
    const reloaded = await loadStore()

    expect(reloaded.$statusbarVisible.get()).toBe(false)
  })
})
