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

  it('does not treat Electron as browser-hosted', () => {
    document.documentElement.dataset.hermesDesktopHost = 'electron'
    expect(isBrowserHostedDesktop()).toBe(false)
  })
})