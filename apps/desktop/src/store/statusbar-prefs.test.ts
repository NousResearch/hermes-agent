import { afterEach, describe, expect, it, vi } from 'vitest'

const STATUSBAR_VISIBLE_STORAGE_KEY = 'hermes.desktop.statusbarVisible'

async function loadStatusbarVisible() {
  vi.resetModules()
  const { $statusbarVisible } = await import('./statusbar-prefs')

  return $statusbarVisible
}

afterEach(() => {
  window.localStorage.removeItem(STATUSBAR_VISIBLE_STORAGE_KEY)
})

describe('statusbar visibility preference', () => {
  it('shows the statusbar when a user has no stored preference', async () => {
    window.localStorage.removeItem(STATUSBAR_VISIBLE_STORAGE_KEY)

    expect((await loadStatusbarVisible()).get()).toBe(true)
  })

  it('preserves an explicit choice to hide the statusbar', async () => {
    window.localStorage.setItem(STATUSBAR_VISIBLE_STORAGE_KEY, 'false')

    expect((await loadStatusbarVisible()).get()).toBe(false)
  })
})
