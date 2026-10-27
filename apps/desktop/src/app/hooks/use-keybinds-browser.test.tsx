import { act, cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { closeTab } = vi.hoisted(() => ({ closeTab: vi.fn() }))

vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: closeTab }))
vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'light', setMode: vi.fn() })
}))

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  closeTab.mockClear()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.document.body.replaceChildren()
  delete window.document.documentElement.dataset.hermesDesktopHost
})

async function mountKeybinds(inTerminal = false) {
  const { useKeybinds } = await import('./use-keybinds')

  const deps = {
    archiveSelectedSession: vi.fn(),
    openNewSessionTab: vi.fn(),
    requestGateway: vi.fn(),
    startFreshSession: vi.fn(),
    toggleCommandCenter: vi.fn(),
    toggleSelectedPin: vi.fn()
  }

  renderHook(() => useKeybinds(deps), {
    wrapper: ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>
  })

  const scope = window.document.createElement('div')

  if (inTerminal) {
    scope.dataset.interactiveTerminal = ''
  }

  const input = window.document.createElement('textarea')
  scope.append(input)
  window.document.body.append(scope)
  input.focus()

  const press = (init: KeyboardEventInit) => {
    const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ctrlKey: true, ...init })
    act(() => { input.dispatchEvent(event) })

    return event
  }

  return { deps, press }
}

it('dispatches browser defaults and remaps from text fields, leaving browser keys and AltGr alone', async () => {
  const { deps, press } = await mountKeybinds()
  const { setBinding } = await import('@/store/keybinds')

  for (const key of ['n', 't', 'w']) {
    expect(press({ key, code: `Key${key.toUpperCase()}` }).defaultPrevented).toBe(false)
  }

  // Windows AltGr reports Ctrl+Alt; Linux can report AltGraph without Ctrl.
  // Polish ń, German @, French AZERTY dead tilde / ʒ are text, not commands.
  for (const ctrlKey of [true, false]) {
    for (const [key, code] of [['ń', 'KeyN'], ['@', 'KeyQ'], ['Dead', 'KeyN'], ['ʒ', 'KeyW']]) {
      const event = press({ key, code, altKey: true, ctrlKey, modifierAltGraph: true })
      expect(event.getModifierState('AltGraph')).toBe(true)
      expect(event.defaultPrevented).toBe(false)
    }
  }

  expect(deps.startFreshSession).not.toHaveBeenCalled()
  expect(deps.openNewSessionTab).not.toHaveBeenCalled()
  expect(closeTab).not.toHaveBeenCalled()

  for (const key of ['F7', 'F8', 'F9']) {
    expect(press({ key, code: key, shiftKey: true }).defaultPrevented).toBe(true)
  }

  expect(deps.startFreshSession).toHaveBeenCalledOnce()
  expect(deps.openNewSessionTab).toHaveBeenCalledOnce()
  expect(closeTab).toHaveBeenCalledOnce()

  act(() => setBinding('session.new', ['mod+shift+f6']))
  expect(press({ key: 'F7', code: 'F7', shiftKey: true }).defaultPrevented).toBe(false)
  expect(press({ key: 'F6', code: 'F6', shiftKey: true }).defaultPrevented).toBe(true)
  expect(deps.startFreshSession).toHaveBeenCalledTimes(2)
})

it('dispatches the macOS ⌃⌘ browser defaults, including ⌃⌘W over an interactive terminal', async () => {
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('MacIntel')
  const { deps, press } = await mountKeybinds(true)

  for (const key of ['n', 't', 'w']) {
    expect(press({ key, code: `Key${key.toUpperCase()}`, metaKey: true, ctrlKey: false }).defaultPrevented).toBe(false)
  }

  for (const key of ['n', 't', 'w']) {
    expect(press({ key, code: `Key${key.toUpperCase()}`, metaKey: true }).defaultPrevented).toBe(true)
  }

  expect(deps.startFreshSession).toHaveBeenCalledOnce()
  expect(deps.openNewSessionTab).toHaveBeenCalledOnce()
  expect(closeTab).toHaveBeenCalledOnce()
})
