import { act, renderHook } from '@testing-library/react'
import type * as ReactRouterDom from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { resetBinding, setBinding } from '@/store/keybinds'

import { useKeybinds } from './use-keybinds'

const mocks = vi.hoisted(() => ({
  closeActiveTerminal: vi.fn(),
  navigate: vi.fn(),
  setMode: vi.fn()
}))

vi.mock('react-router-dom', async importOriginal => ({
  ...(await importOriginal<typeof ReactRouterDom>()),
  useNavigate: () => mocks.navigate
}))

vi.mock('@/app/right-sidebar/terminal/terminals', () => ({
  closeActiveTerminal: mocks.closeActiveTerminal,
  createTerminal: vi.fn(),
  cycleTerminal: vi.fn()
}))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'light', setMode: mocks.setMode })
}))

const deps = {
  openNewSessionTab: vi.fn(),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

function focusedTerminal(interactive: boolean): HTMLTextAreaElement {
  const terminal = window.document.createElement('div')
  terminal.dataset.terminal = ''

  if (interactive) {
    terminal.dataset.interactiveTerminal = ''
  }

  const input = window.document.createElement('textarea')
  terminal.append(input)
  window.document.body.append(terminal)
  input.focus()

  return input
}

function pressCtrlW(target: HTMLElement, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {
    bubbles: true,
    cancelable: true,
    code: 'KeyW',
    ctrlKey: true,
    key: 'w',
    ...init
  })

  act(() => target.dispatchEvent(event))

  return event
}

describe('terminal close-tab keybind ownership', () => {
  beforeEach(() => {
    setBinding('view.closeTab', ['ctrl+w'])
  })

  afterEach(() => {
    resetBinding('view.closeTab')
    mocks.closeActiveTerminal.mockReset()
    mocks.navigate.mockReset()
    mocks.setMode.mockReset()
    window.document.body.replaceChildren()
  })

  it('leaves Ctrl+W to an interactive terminal', () => {
    renderHook(() => useKeybinds(deps))

    const event = pressCtrlW(focusedTerminal(true))

    expect(event.defaultPrevented).toBe(false)
    expect(mocks.closeActiveTerminal).not.toHaveBeenCalled()
  })

  it('keeps close-tab behavior for a read-only agent terminal', () => {
    renderHook(() => useKeybinds(deps))

    const event = pressCtrlW(focusedTerminal(false))

    expect(event.defaultPrevented).toBe(true)
    expect(mocks.closeActiveTerminal).toHaveBeenCalledOnce()
  })

  it('keeps a rebound close-tab chord in an interactive terminal', () => {
    setBinding('view.closeTab', ['ctrl+alt+w'])
    renderHook(() => useKeybinds(deps))

    const event = pressCtrlW(focusedTerminal(true), { altKey: true })

    expect(event.defaultPrevented).toBe(true)
    expect(mocks.closeActiveTerminal).toHaveBeenCalledOnce()
  })
})
