import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.sidebarNavHidden.v1'

const loadStore = () => import('./sidebar-nav-visibility')

describe('sidebar navigation visibility', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('persists hidden rows and restores a row without retaining an empty preference', async () => {
    const first = await loadStore()

    first.setSidebarNavItemVisible('artifacts', false)
    first.setSidebarNavItemVisible('plugin:kanban', false)
    expect(first.$sidebarNavHiddenIds.get()).toEqual(['artifacts', 'plugin:kanban'])

    vi.resetModules()
    const reloaded = await loadStore()
    expect(reloaded.$sidebarNavHiddenIds.get()).toEqual(['artifacts', 'plugin:kanban'])

    reloaded.setSidebarNavItemVisible('artifacts', true)
    reloaded.setSidebarNavItemVisible('plugin:kanban', true)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })
})
