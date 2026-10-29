import { beforeEach, describe, expect, it, vi } from 'vitest'

const loadStore = async () => {
  vi.resetModules()

  return import('./sidebar-project-display')
}

describe('sidebar project display preferences (#137443)', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('keeps the existing sidebar presentation by default and persists choices', async () => {
    const first = await loadStore()

    expect(first.$showEmptyWorktreeLanes.get()).toBe(true)
    expect(first.$projectLabelMode.get()).toBe('short')

    first.setShowEmptyWorktreeLanes(false)
    first.setProjectLabelMode('full')

    expect(window.localStorage.getItem('hermes.desktop.sidebar.showEmptyWorktreeLanes')).toBe('false')
    expect(window.localStorage.getItem('hermes.desktop.sidebar.projectLabelMode')).toBe('full')
  })

  it('falls back to the safe defaults for malformed stored values', async () => {
    window.localStorage.setItem('hermes.desktop.sidebar.showEmptyWorktreeLanes', 'not-a-boolean')
    window.localStorage.setItem('hermes.desktop.sidebar.projectLabelMode', 'everywhere')

    const store = await loadStore()

    expect(store.$showEmptyWorktreeLanes.get()).toBe(true)
    expect(store.$projectLabelMode.get()).toBe('short')
  })
})
