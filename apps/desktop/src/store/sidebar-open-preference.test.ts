import { beforeEach, describe, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.sidebarSessionsOpenInNewTab'

describe('$sidebarSessionsOpenInNewTab', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('defaults to tab-first without overwriting absent storage', async () => {
    const { $sidebarSessionsOpenInNewTab } = await import('./sidebar-open-preference')

    expect($sidebarSessionsOpenInNewTab.get()).toBe(true)
    expect(window.localStorage.getItem(KEY)).toBeNull()
  })

  it('preserves an explicitly persisted replace-main preference', async () => {
    window.localStorage.setItem(KEY, 'false')

    const { $sidebarSessionsOpenInNewTab } = await import('./sidebar-open-preference')

    expect($sidebarSessionsOpenInNewTab.get()).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBe('false')
  })

  it('persists false when the user chooses replace-main', async () => {
    const { $sidebarSessionsOpenInNewTab } = await import('./sidebar-open-preference')

    $sidebarSessionsOpenInNewTab.set(false)

    expect(window.localStorage.getItem(KEY)).toBe('false')
  })
})
