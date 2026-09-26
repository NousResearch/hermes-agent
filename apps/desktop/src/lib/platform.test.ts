// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

const platform = vi.hoisted(() => ({ value: 'MacIntel' }))

vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', {
    configurable: true,
    get: () => platform.value
  })
})

import { isBrowserHostedDesktop } from './platform'

describe('isBrowserHostedDesktop', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute('data-hermes-desktop-host')
    delete (window as Window & { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__
    delete (window as Window & { __HERMES_AUTH_REQUIRED__?: boolean }).__HERMES_AUTH_REQUIRED__
    delete (window as Window & { __HERMES_UI_SURFACE__?: string }).__HERMES_UI_SURFACE__
  })

  it('recognizes the browser-hosted renderer marker', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    expect(isBrowserHostedDesktop()).toBe(true)
  })

  it('recognizes browser bootstrap globals before the bridge marker is written', () => {
    const win = window as Window & { __HERMES_SESSION_TOKEN__?: string }
    win.__HERMES_SESSION_TOKEN__ = 'test-token'
    expect(isBrowserHostedDesktop()).toBe(true)
    delete win.__HERMES_SESSION_TOKEN__
  })

  // The Webapp surface is served without an injected token; the bridge still
  // installs for it, so the pre-marker check must agree.
  it('recognizes the tokenless Webapp surface before the bridge marker is written', () => {
    const win = window as Window & { __HERMES_UI_SURFACE__?: string }
    win.__HERMES_UI_SURFACE__ = 'webapp'
    expect(isBrowserHostedDesktop()).toBe(true)
  })

  it('does not treat Electron as browser-hosted', () => {
    document.documentElement.dataset.hermesDesktopHost = 'electron'
    expect(isBrowserHostedDesktop()).toBe(false)
  })
})