import { beforeEach, describe, expect, it, vi } from 'vitest'

// The mode atom resolves from storage at import time, so every scenario resets
// the module graph and re-imports the store after seeding localStorage.
async function loadToolViewStore() {
  return await import('@/store/tool-view')
}

describe('tool view mode', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('defaults to product when nothing is persisted', async () => {
    const store = await loadToolViewStore()

    expect(store.$toolViewMode.get()).toBe('product')
  })

  it('migrates the legacy technical boolean', async () => {
    window.localStorage.setItem('hermes.desktop.toolView.technical', 'true')

    const store = await loadToolViewStore()

    expect(store.$toolViewMode.get()).toBe('technical')
  })

  it('reads the persisted mode string, including cards', async () => {
    window.localStorage.setItem('hermes.desktop.toolView.mode', 'cards')

    const store = await loadToolViewStore()

    expect(store.$toolViewMode.get()).toBe('cards')
  })

  it('ignores an unknown persisted mode', async () => {
    window.localStorage.setItem('hermes.desktop.toolView.mode', 'bogus')

    const store = await loadToolViewStore()

    expect(store.$toolViewMode.get()).toBe('product')
  })

  it('persists setToolViewMode under the new string key and clears it on product', async () => {
    const store = await loadToolViewStore()

    store.setToolViewMode('cards')
    expect(window.localStorage.getItem('hermes.desktop.toolView.mode')).toBe('cards')

    store.setToolViewMode('product')
    expect(window.localStorage.getItem('hermes.desktop.toolView.mode')).toBe('product')
  })
})
