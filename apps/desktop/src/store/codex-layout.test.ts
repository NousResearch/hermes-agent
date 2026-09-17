import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.codexLayout.v1'

const loadStore = () => import('./codex-layout')

describe('Codex layout preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('defaults on and restores both persisted toggle states after reload', async () => {
    let store = await loadStore()

    expect(store.$codexLayout.get()).toBe(true)

    store.setCodexLayout(false)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('off')

    vi.resetModules()
    store = await loadStore()
    expect(store.$codexLayout.get()).toBe(false)

    store.setCodexLayout(true)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('on')

    vi.resetModules()
    store = await loadStore()
    expect(store.$codexLayout.get()).toBe(true)
  })
})
