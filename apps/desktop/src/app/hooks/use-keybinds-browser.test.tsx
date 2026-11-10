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
  vi.useRealTimers()
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
    act(() => {
      input.dispatchEvent(event)
    })

    return event
  }

  return { deps, press }
}

it('dispatches the Alt defaults and remaps from text fields, leaving browser keys and AltGr alone', async () => {
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
  const { deps, press } = await mountKeybinds()
  const { setBinding } = await import('@/store/keybinds')

  for (const key of ['n', 't', 'w']) {
    expect(press({ key, code: `Key${key.toUpperCase()}` }).defaultPrevented).toBe(false)
  }

  // Windows AltGr reports Ctrl+Alt; Linux can report AltGraph without Ctrl.
  // Polish ń, German @, French AZERTY dead tilde / ʒ are text, not commands.
  for (const ctrlKey of [true, false]) {
    for (const [key, code] of [
      ['ń', 'KeyN'],
      ['@', 'KeyQ'],
      ['Dead', 'KeyN'],
      ['ʒ', 'KeyW']
    ]) {
      const event = press({ key, code, altKey: true, ctrlKey, modifierAltGraph: true })
      expect(event.getModifierState('AltGraph')).toBe(true)
      expect(event.defaultPrevented).toBe(false)
    }
  }

  expect(deps.startFreshSession).not.toHaveBeenCalled()
  expect(deps.openNewSessionTab).not.toHaveBeenCalled()
  expect(closeTab).not.toHaveBeenCalled()

  for (const key of ['n', 't', 'w']) {
    expect(press({ key, code: `Key${key.toUpperCase()}`, altKey: true, ctrlKey: false }).defaultPrevented).toBe(true)
  }

  expect(deps.startFreshSession).toHaveBeenCalledOnce()
  expect(deps.openNewSessionTab).toHaveBeenCalledOnce()
  expect(closeTab).toHaveBeenCalledOnce()

  act(() => setBinding('session.new', ['mod+shift+f6']))
  expect(press({ key: 'n', code: 'KeyN', altKey: true, ctrlKey: false }).defaultPrevented).toBe(false)
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

it('keeps an Option remap on macOS, where Firefox reports Option as AltGraph', async () => {
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('MacIntel')
  const { press } = await mountKeybinds()
  const { setBinding } = await import('@/store/keybinds')
  const { $commandPaletteOpen } = await import('@/store/command-palette')
  act(() => setBinding('nav.commandPalette', ['alt+.']))

  const option = press({ key: '≥', code: 'Period', altKey: true, ctrlKey: false, modifierAltGraph: true })

  expect(option.getModifierState('AltGraph')).toBe(true)
  expect(option.defaultPrevented).toBe(true)
  expect($commandPaletteOpen.get()).toBe(true)
})

it('steps the session switcher with Alt+] and commits when Alt is released', async () => {
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
  const { $selectedStoredSessionId, $sessions } = await import('@/store/session')
  const { $switcherOpen, switcherActive } = await import('@/store/session-switcher')
  $sessions.set(['a', 'b', 'c'].map(id => ({ id }) as never))
  $selectedStoredSessionId.set('a')
  const { press } = await mountKeybinds()

  const release = (key: string, init: KeyboardEventInit = {}) =>
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key, ...init }))
    })

  expect(press({ key: ']', code: 'BracketRight', altKey: true, ctrlKey: false }).defaultPrevented).toBe(true)
  expect(press({ key: ']', code: 'BracketRight', altKey: true, ctrlKey: false }).defaultPrevented).toBe(true)
  expect($switcherOpen.get()).toBe(true)

  release(']', { altKey: true })
  expect($switcherOpen.get()).toBe(true)
  release('Alt')
  expect($switcherOpen.get()).toBe(false)
  expect(switcherActive()).toBe(false)
})

it('keeps the switcher step held while another key is released', async () => {
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
  const { $selectedStoredSessionId, $sessions } = await import('@/store/session')
  const { $switcherOpen, SWITCHER_REVEAL_MS } = await import('@/store/session-switcher')
  $sessions.set(['a', 'b', 'c'].map(id => ({ id }) as never))
  $selectedStoredSessionId.set('a')
  const { press } = await mountKeybinds()
  vi.useFakeTimers()

  const release = (key: string, code: string, init: KeyboardEventInit = {}) =>
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, code, key, ...init }))
    })

  press({ key: ']', code: 'BracketRight', altKey: true, ctrlKey: false })
  release('a', 'KeyA', { altKey: true })
  act(() => vi.advanceTimersByTime(SWITCHER_REVEAL_MS))
  expect($switcherOpen.get()).toBe(true)

  release(']', 'BracketRight', { altKey: true })
  release('Alt', 'AltLeft')
  expect($switcherOpen.get()).toBe(false)

  // A quick tap of the step key itself, Alt still down: no HUD.
  press({ key: ']', code: 'BracketRight', altKey: true, ctrlKey: false })
  release(']', 'BracketRight', { altKey: true })
  act(() => vi.advanceTimersByTime(SWITCHER_REVEAL_MS))
  expect($switcherOpen.get()).toBe(false)
})

it('marks only chords an action ran, so the terminal keeps a declined one', async () => {
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
  const { deps, press } = await mountKeybinds(true)
  const { setBinding } = await import('@/store/keybinds')
  const { keybindHandled } = await import('@/lib/keybinds/handled-keys')

  const ran = press({ key: 't', code: 'KeyT', altKey: true, ctrlKey: false })
  expect(deps.openNewSessionTab).toHaveBeenCalledOnce()
  expect(keybindHandled(ran)).toBe(true)

  // view.tabSlot.9 declines with no tab strip: cancelled, but no action ran.
  act(() => setBinding('view.tabSlot.9', ['alt+9']))

  const declined = new KeyboardEvent('keydown', {
    altKey: true,
    bubbles: true,
    cancelable: true,
    code: 'Digit9',
    key: '9'
  })

  act(() => {
    window.document.body.dispatchEvent(declined)
  })
  expect(declined.defaultPrevented).toBe(true)
  expect(keybindHandled(declined)).toBe(false)
})
