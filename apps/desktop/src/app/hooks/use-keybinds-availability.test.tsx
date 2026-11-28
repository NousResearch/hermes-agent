import { act, cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'light', setMode: vi.fn() })
}))

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

async function mountKeybinds() {
  const { useKeybinds } = await import('./use-keybinds')

  renderHook(
    () =>
      useKeybinds({
        archiveSelectedSession: vi.fn(),
        openNewSessionTab: vi.fn(),
        requestGateway: vi.fn(),
        startFreshSession: vi.fn(),
        toggleCommandCenter: vi.fn(),
        toggleSelectedPin: vi.fn()
      }),
    { wrapper: ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter> }
  )

  // ⌘⇧S / Ctrl+Shift+S: the statusbar toggle's default chord.
  return () => {
    const event = new KeyboardEvent('keydown', {
      bubbles: true,
      cancelable: true,
      code: 'KeyS',
      ctrlKey: true,
      key: 'S',
      shiftKey: true
    })

    act(() => {
      window.document.body.dispatchEvent(event)
    })

    return event
  }
}

it('dispatches the statusbar chord to the active shell without mutating the other shell', async () => {
  const pressToggle = await mountKeybinds()
  const { $portalStatusBarOpen, setPortalStatusBarOpen } = await import('@/store/portal-status-bar')
  const { $statusbarVisible } = await import('@/store/statusbar-prefs')
  const { $shellChoice } = await import('@/store/shell-variant')

  setPortalStatusBarOpen(true)

  // In the redesign, ⌘⇧S toggles the redesign status bar while leaving classic's strip intact.
  expect(pressToggle().defaultPrevented).toBe(true)
  expect($portalStatusBarOpen.get()).toBe(false)
  expect($statusbarVisible.get()).toBe(true)

  act(() => $shellChoice.set(false))

  expect(pressToggle().defaultPrevented).toBe(true)
  expect($statusbarVisible.get()).toBe(false)
})
