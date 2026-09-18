import { beforeEach, describe, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.sidebarSessionsOpenInNewTab'

describe('$sidebarSessionsOpenInNewTab', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('defaults to false without writing storage on read', async () => {
    const { $sidebarSessionsOpenInNewTab } = await import('./sidebar-open-preference')

    expect($sidebarSessionsOpenInNewTab.get()).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBeNull()
  })

  it('falls back to false but leaves malformed storage untouched until a write', async () => {
    window.localStorage.setItem(KEY, 'not-a-bool')

    const { $sidebarSessionsOpenInNewTab } = await import('./sidebar-open-preference')

    expect($sidebarSessionsOpenInNewTab.get()).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBe('not-a-bool')
  })

  it('persists true when the user enables it', async () => {
    const { $sidebarSessionsOpenInNewTab } = await import('./sidebar-open-preference')

    $sidebarSessionsOpenInNewTab.set(true)

    expect(window.localStorage.getItem(KEY)).toBe('true')
  })
})
