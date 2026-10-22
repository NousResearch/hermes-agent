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

describe('statusbar hidden items', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('surfaces the approval pill for installs that hid it under the v1 defaults, keeping their other choices', async () => {
    window.localStorage.setItem(
      'hermes.desktop.statusbarHidden',
      JSON.stringify(['approval-mode', 'cron', 'gateway-health'])
    )

    const { $statusbarHiddenIds } = await loadStore()

    expect($statusbarHiddenIds.get()).toEqual(['cron', 'gateway-health', 'session-cost'])
  })

  it('still honours hiding the approval pill after the update', async () => {
    const first = await loadStore()

    first.setStatusbarItemVisible('approval-mode', false)

    vi.resetModules()
    const reloaded = await loadStore()

    expect(reloaded.$statusbarHiddenIds.get()).toContain('approval-mode')
  })

  it('seeds the spend chip hidden for a v2 store, which never had a chance to choose it', async () => {
    window.localStorage.setItem('hermes.desktop.statusbarHidden.v2', JSON.stringify(['cron', 'gateway-health']))

    const { $statusbarHiddenIds } = await loadStore()

    // The user's own choices survive; the chip they never saw is hidden rather
    // than arriving as a cost readout on update.
    expect($statusbarHiddenIds.get()).toEqual(['cron', 'gateway-health', 'session-cost'])
  })

  it('keeps a v3 store choice to show the spend chip', async () => {
    const first = await loadStore()

    first.setStatusbarItemVisible('session-cost', true)

    vi.resetModules()
    const reloaded = await loadStore()

    expect(reloaded.$statusbarHiddenIds.get()).not.toContain('session-cost')
  })
})
