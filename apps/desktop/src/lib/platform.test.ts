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
  })

  it('recognizes the browser-hosted renderer marker', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    expect(isBrowserHostedDesktop()).toBe(true)
  })

  it('does not treat Electron as browser-hosted', () => {
    document.documentElement.dataset.hermesDesktopHost = 'electron'
    expect(isBrowserHostedDesktop()).toBe(false)
  })
})