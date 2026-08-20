import { beforeEach, describe, expect, it, vi } from 'vitest'

async function loadSidebarStores() {
  const layout = await import('./layout')
  const views = await import('./sidebar-views')

  return { layout, views }
}

describe('saved sidebar view persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('restores saved views through a fresh module load', async () => {
    const first = await loadSidebarStores()
    first.layout.setSidebarGrouping('none')
    first.layout.toggleSidebarStatusFilter('working')
    first.views.saveCurrentSidebarView('Working now', { id: 'working-now', now: 100 })

    vi.resetModules()
    const reloaded = await loadSidebarStores()

    expect(reloaded.views.$savedSidebarViews.get().views).toEqual([
      expect.objectContaining({
        id: 'working-now',
        name: 'Working now',
        state: expect.objectContaining({
          grouping: 'none',
          filters: expect.objectContaining({ statuses: ['working'] })
        })
      })
    ])
  })
})
