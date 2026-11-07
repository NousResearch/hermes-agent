import { renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setBinding } from '@/store/keybinds'

import { useKeybinds } from './use-keybinds'

const deps = {
  archiveSelectedSession: vi.fn(),
  isActiveRunBusy: vi.fn(),
  openNewSessionTab: vi.fn(),
  requestGateway: vi.fn(),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

/** Focus an input inside a terminal matching the production DOM shape:
 * [data-terminal] carries [data-interactive-terminal] only on the user PTY. */
function focusedTerminal(): HTMLTextAreaElement {
  const terminal = window.document.createElement('div')
  terminal.dataset.terminal = ''
  terminal.dataset.interactiveTerminal = ''

  const input = window.document.createElement('textarea')
  terminal.append(input)
  window.document.body.append(terminal)
  input.focus()

  return input
}

/** The Ctrl+C chord as comboFromEvent canonicalizes it off macOS (jsdom's
 * platform is not mac), where `ctrl` folds to `mod` — the same shape the
 * shipped `ctrl+c` default resolves to. */
function pressStopChord(target: HTMLElement): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {
    bubbles: true,
    cancelable: true,
    code: 'KeyC',
    ctrlKey: true,
    key: 'c'
  })

  target.dispatchEvent(event)

  return event
}

describe('session.stop (Ctrl+C) keybind', () => {
  let unmount: (() => void) | undefined
  let stopEvents: number

  beforeEach(() => {
    vi.mocked(deps.isActiveRunBusy).mockReturnValue(false)
    setBinding('session.stop', ['ctrl+c'])
    stopEvents = 0
    window.addEventListener('hermes:stop-active-run', () => {
      stopEvents += 1
    })
    unmount = renderHook(() => useKeybinds(deps), {
      wrapper: ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>
    }).unmount
  })

  afterEach(() => {
    unmount?.()
    window.document.body.innerHTML = ''
  })

  it('stops the run while a turn is running', () => {
    vi.mocked(deps.isActiveRunBusy).mockReturnValue(true)

    const event = pressStopChord(window.document.body)

    expect(event.defaultPrevented).toBe(true)
    expect(stopEvents).toBe(1)
  })

  it('falls through while idle — the chord keeps its native meaning', () => {
    const event = pressStopChord(window.document.body)

    expect(event.defaultPrevented).toBe(false)
    expect(stopEvents).toBe(0)
  })

  it('yields to copy when the user has a text selection', () => {
    vi.mocked(deps.isActiveRunBusy).mockReturnValue(true)

    const input = window.document.createElement('textarea')
    window.document.body.append(input)
    input.value = 'selected text'
    input.focus()
    input.setSelectionRange(0, 4)

    const event = pressStopChord(input)

    expect(event.defaultPrevented).toBe(false)
    expect(stopEvents).toBe(0)
  })

  it('yields to the PTY when a user terminal owns focus', () => {
    vi.mocked(deps.isActiveRunBusy).mockReturnValue(true)

    const input = focusedTerminal()

    const event = pressStopChord(input)

    expect(event.defaultPrevented).toBe(false)
    expect(stopEvents).toBe(0)
  })
})
