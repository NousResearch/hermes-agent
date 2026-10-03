// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const platform = vi.hoisted(() => ({ value: 'Win32' }))

vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', {
    configurable: true,
    get: () => platform.value
  })
})

async function loadTranslucency(host: 'browser' | 'electron') {
  document.documentElement.dataset.hermesDesktopHost = host
  vi.resetModules()
  return import('@/store/translucency')
}

describe('browser translucency capability', () => {
  beforeEach(() => {
    localStorage.clear()
    document.documentElement.removeAttribute('data-hermes-desktop-host')
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.resetModules()
  })

  it('disables native window translucency in the browser host even on Windows', async () => {
    const translucency = await loadTranslucency('browser')

    expect(translucency.GLASS_SUPPORTED).toBe(false)
    expect(translucency.TRANSLUCENCY_SUPPORTED).toBe(false)
  })

  it('retains native capability detection outside the browser host', async () => {
    const translucency = await loadTranslucency('electron')

    expect(translucency.TRANSLUCENCY_SUPPORTED).toBe(true)
  })
})