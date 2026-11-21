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

it('lets an action this window does not offer pass its chord through, and dispatches it after a flip', async () => {
  const pressToggle = await mountKeybinds()
  const { $statusbarVisible } = await import('@/store/statusbar-prefs')
  const { $shellChoice } = await import('@/store/shell-variant')

  // The redesign has no statusbar strip: the chord is not swallowed as a silent no-op.
  expect(pressToggle().defaultPrevented).toBe(false)
  expect($statusbarVisible.get()).toBe(true)

  act(() => $shellChoice.set(false))

  expect(pressToggle().defaultPrevented).toBe(true)
  expect($statusbarVisible.get()).toBe(false)
})
