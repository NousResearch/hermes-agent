import { afterEach, describe, expect, it, vi } from 'vitest'

const GROUPED_KEY = 'hermes.desktop.agentsGroupedByWorkspace'

async function freshLayoutStore() {
  vi.resetModules()

  return await import('./layout')
}

afterEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

describe('desktop sidebar session visibility', () => {
  it('does not restore persisted Projects mode over the human Sessions inbox', async () => {
    window.localStorage.setItem(GROUPED_KEY, 'true')

    const { $sidebarAgentsGrouped } = await freshLayoutStore()

    expect($sidebarAgentsGrouped.get()).toBe(false)
  })

  it('keeps Projects mode as per-window state instead of persisting it across reloads', async () => {
    const first = await freshLayoutStore()

    first.setSidebarAgentsGrouped(true)

    expect(first.$sidebarAgentsGrouped.get()).toBe(true)
    expect(window.localStorage.getItem(GROUPED_KEY)).toBeNull()

    const second = await freshLayoutStore()

    expect(second.$sidebarAgentsGrouped.get()).toBe(false)
  })
})
