// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BrowserHostGlobals } from './platform'

const win = window as Window & BrowserHostGlobals

beforeEach(() => {
  vi.resetModules()
  delete document.documentElement.dataset.hermesDesktopHost
  delete (window as Partial<Window>).hermesDesktop
  delete win.__HERMES_UI_SURFACE__
})

afterEach(() => {
  delete document.documentElement.dataset.hermesDesktopHost
  delete (window as Partial<Window>).hermesDesktop
  delete win.__HERMES_UI_SURFACE__
})

describe('Clerk host configuration', () => {
  it('enables the browser provider for the browser-hosted renderer', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    window.hermesDesktop = {} as Window['hermesDesktop']
    const config = await import('./clerk-config')

    expect(config.clerkBrowserEnabled).toBe(true)
    expect(config.clerkEnabled).toBe(false)
  })

  it('detects the browser host before its bridge is installed', async () => {
    win.__HERMES_UI_SURFACE__ = 'webapp'
    const config = await import('./clerk-config')

    expect(config.clerkBrowserEnabled).toBe(true)
    expect(config.clerkEnabled).toBe(false)
  })

  it.each([true, false])('keeps the browser provider off for a native bridge (enabled=%s)', async enabled => {
    window.hermesDesktop = { clerk: { enabled } } as Window['hermesDesktop']
    const config = await import('./clerk-config')

    expect(config.clerkBrowserEnabled).toBe(false)
    expect(config.clerkEnabled).toBe(enabled)
  })

  it('does not enable the browser provider on an unrecognized host', async () => {
    const config = await import('./clerk-config')

    expect(config.clerkBrowserEnabled).toBe(false)
    expect(config.clerkEnabled).toBe(false)
  })
})
