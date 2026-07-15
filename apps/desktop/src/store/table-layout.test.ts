import { beforeEach, describe, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.tableLayout.v1'

async function loadStore() {
  vi.resetModules()

  return import('./table-layout')
}

describe('Markdown table layout store', () => {
  beforeEach(() => {
    window.localStorage.clear()
    delete document.documentElement.dataset.hermesTableLayout
  })

  it('defaults to the current fit behavior', async () => {
    const { $tableLayout } = await loadStore()

    expect($tableLayout.get()).toBe('fit')
    expect(document.documentElement.dataset.hermesTableLayout).toBe('fit')
  })

  it('hydrates and applies the persisted scroll behavior', async () => {
    window.localStorage.setItem(KEY, 'scroll')

    const { $tableLayout } = await loadStore()

    expect($tableLayout.get()).toBe('scroll')
    expect(document.documentElement.dataset.hermesTableLayout).toBe('scroll')
  })

  it('persists layout changes', async () => {
    const { setTableLayout } = await loadStore()

    setTableLayout('scroll')

    expect(window.localStorage.getItem(KEY)).toBe('scroll')
    expect(document.documentElement.dataset.hermesTableLayout).toBe('scroll')
  })

  it('falls back safely when the persisted value is unknown', async () => {
    window.localStorage.setItem(KEY, 'overflow')

    const { $tableLayout } = await loadStore()

    expect($tableLayout.get()).toBe('fit')
    expect(document.documentElement.dataset.hermesTableLayout).toBe('fit')
  })
})
