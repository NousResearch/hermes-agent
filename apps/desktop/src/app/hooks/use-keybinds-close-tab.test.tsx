import { renderHook } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { isMacPlatform } from '@/lib/platform'
import { resetBinding, setBinding } from '@/store/keybinds'

import { useKeybinds } from './use-keybinds'

const mocks = vi.hoisted(() => ({
  closeActiveTerminal: vi.fn(),
  setMode: vi.fn()
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
  archiveSelectedSession: vi.fn(),
  openNewSessionTab: vi.fn(),
  requestGateway: vi.fn(),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

/** Focus an input inside a terminal matching the production DOM shape
 * (instance.tsx): the [data-terminal] scope carries
 * [data-interactive-terminal] only on the user PTY. */
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

function pressCloseChord(target: HTMLElement, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {
    bubbles: true,
    cancelable: true,
    code: 'KeyW',
    ctrlKey: !isMacPlatform(),
    key: 'w',
    metaKey: isMacPlatform(),
    ...init
  })

  target.dispatchEvent(event)

  return event
}

describe('close-tab chord over a focused terminal', () => {
  let unmount: (() => void) | undefined

  beforeAll(() => {
    // The closeTerminal handler asks the live tree whether the terminal pane
    // is on screen; give it one with the terminal active.
    registry.register({ id: 'terminal', area: 'panes', title: 'terminal', render: () => null })
  })

  beforeEach(() => {
    $layoutTree.set(split('row', [group(['workspace']), group(['terminal'])]))
    setBinding('view.closeTab', ['mod+w'])
    unmount = renderHook(() => useKeybinds(deps), {
      wrapper: ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>
    }).unmount
  })

  afterEach(() => {
    unmount?.()
    unmount = undefined
    $layoutTree.set(null)
    resetBinding('view.closeTab')
    mocks.closeActiveTerminal.mockReset()
    window.document.body.replaceChildren()
  })

  it('leaves the default chord to an interactive terminal (word erase)', () => {
    const event = pressCloseChord(focusedTerminal(true))

    // Not claimed: xterm sees the keystroke and writes ^W to the PTY.
    expect(event.defaultPrevented).toBe(false)
    expect(mocks.closeActiveTerminal).not.toHaveBeenCalled()
  })

  it('keeps close-tab for a read-only agent terminal', () => {
    const event = pressCloseChord(focusedTerminal(false))

    expect(event.defaultPrevented).toBe(true)
    expect(mocks.closeActiveTerminal).toHaveBeenCalledOnce()
  })

  it('keeps Ctrl/Cmd+Shift+W as the explicit close chord', () => {
    const event = pressCloseChord(focusedTerminal(true), { key: 'W', shiftKey: true })

    expect(event.defaultPrevented).toBe(true)
    expect(mocks.closeActiveTerminal).toHaveBeenCalledOnce()
  })

  it('keeps a rebound close-tab chord closing in an interactive terminal', () => {
    // The chord was rebound to alt+W: only the platform's bare close chord
    // (mod+w) is the terminal's word erase.
    setBinding('view.closeTab', ['mod+alt+w'])
    const event = pressCloseChord(focusedTerminal(true), { altKey: true })

    expect(event.defaultPrevented).toBe(true)
    expect(mocks.closeActiveTerminal).toHaveBeenCalledOnce()
  })

  it('keeps closing non-terminal surfaces', () => {
    const input = window.document.createElement('textarea')
    window.document.body.append(input)
    input.focus()

    const event = pressCloseChord(input)

    expect(event.defaultPrevented).toBe(true)
    expect(mocks.closeActiveTerminal).not.toHaveBeenCalled()
  })
})
