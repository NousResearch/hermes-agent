import { afterEach, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.sidebarHoverReveal'

afterEach(() => {
  localStorage.clear()
  vi.resetModules()
})

it('defaults to hover reveal without an initialization write and restores either explicit choice after a reload', async () => {
  localStorage.clear()
  vi.resetModules()
  const initial = await import('./sidebar-hover-reveal')
  expect(initial.$sidebarHoverReveal.get()).toBe(true)
  expect(localStorage.getItem(KEY)).toBeNull()

  // A fresh module load represents a restarted renderer reading the same storage.
  for (const choice of [false, true]) {
    initial.setSidebarHoverReveal(choice)
    expect(localStorage.getItem(KEY)).toBe(String(choice))

    vi.resetModules()
    const restored = await import('./sidebar-hover-reveal')
    expect(restored.$sidebarHoverReveal.get()).toBe(choice)
    expect(localStorage.getItem(KEY)).toBe(String(choice))
  }
})
