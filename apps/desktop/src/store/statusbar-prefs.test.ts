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

describe('statusbar item order', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('keeps a custom order while leaving newly added items in their default position', async () => {
    const { $statusbarItemOrder, orderStatusbarItems, setStatusbarItemOrder } = await loadStore()

    expect(
      orderStatusbarItems([{ id: 'workspace' }, { id: 'gateway' }, { id: 'timer' }], ['timer', 'gateway'])
    ).toEqual([{ id: 'workspace' }, { id: 'timer' }, { id: 'gateway' }])

    setStatusbarItemOrder('right', ['timer', 'gateway'])
    expect($statusbarItemOrder.get()).toEqual({ left: [], right: ['timer', 'gateway'] })

    vi.resetModules()
    expect((await loadStore()).$statusbarItemOrder.get()).toEqual({ left: [], right: ['timer', 'gateway'] })
  })
})

describe('statusbar order fallback', () => {
  it('keeps non-menu controls and new items in their shipped slots', async () => {
    const { orderStatusbarItems } = await loadStore()
    const ids = ['command-center', 'gateway-switcher', 'profile-switcher', 'plugin-new', 'workspace', 'session']
    const items = ids.map(id => ({ id }))
    expect(orderStatusbarItems(items, ['session', 'command-center', 'workspace']).map(item => item.id)).toEqual([
      'session',
      'gateway-switcher',
      'profile-switcher',
      'plugin-new',
      'command-center',
      'workspace'
    ])
  })
})
