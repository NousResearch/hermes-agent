import { beforeEach, describe, expect, it, vi } from 'vitest'

const LEGACY_VISIBLE_KEY = 'hermes.desktop.statusbarVisible'

const loadStore = () => import('./statusbar-prefs')

const SHELL_KEY = 'hermes.desktop.agentShell.v1'

describe('statusbar whole-bar visibility', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.localStorage.setItem(SHELL_KEY, 'false')
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

  it('has no strip in the redesign, so no toggle there can change the classic preference', async () => {
    const store = await loadStore()
    const { setPortalPreferred } = await import('./shell-variant')
    expect(store.$statusbarShown.get()).toBe(true)

    setPortalPreferred(true)
    expect((await import('./statusbar-strip')).$statusbarStripAvailable.get()).toBe(false)
    expect(store.$statusbarShown.get()).toBe(false)
    store.toggleStatusbarVisible()
    expect(store.$statusbarVisible.get()).toBe(true)

    setPortalPreferred(false)
    expect(store.$statusbarShown.get()).toBe(true)
    store.toggleStatusbarVisible()
    expect(store.$statusbarShown.get()).toBe(false)
  })

  it('keeps the Simple-mode policy: the bar rests hidden unless the free tier Sign in needs it', async () => {
    const store = await loadStore()
    const mode = await import('./interface-mode')
    mode.setInterfaceMode('simple')
    expect(store.$statusbarShown.get()).toBe(false)
    mode.setModeContext({ freeTierSignInOpen: true })
    expect(store.$statusbarShown.get()).toBe(true)
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

    expect($statusbarHiddenIds.get()).toEqual(['cron', 'gateway-health'])
  })

  it('still honours hiding the approval pill after the update', async () => {
    const first = await loadStore()

    first.setStatusbarItemVisible('approval-mode', false)

    vi.resetModules()
    const reloaded = await loadStore()

    expect(reloaded.$statusbarHiddenIds.get()).toContain('approval-mode')
  })
})
