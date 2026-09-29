import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  commandFocusedTerminal,
  registerTerminalContextMenu,
  terminalMenuHandleFor,
  wordEraseFocusedTerminal
} from './terminal-context-menu'

const handle = (overrides: Partial<Parameters<typeof registerTerminalContextMenu>[1]> = {}) => ({
  getSelection: () => '',
  paste: null,
  reload: vi.fn(),
  selectAll: vi.fn(),
  wordErase: null,
  ...overrides
})

afterEach(() => {
  document.body.innerHTML = ''
})

/** instance.tsx shape: outer div carries data-terminal, hostRef points at the
 *  inner xterm host div, and a click lands on the canvas inside it. */
function mountProductionShape(): { canvas: HTMLElement; host: HTMLElement; scope: HTMLElement } {
  const scope = document.createElement('div')
  scope.setAttribute('data-terminal', '')
  const host = document.createElement('div')
  const canvas = document.createElement('canvas')
  host.appendChild(canvas)
  scope.appendChild(host)
  document.body.appendChild(scope)

  return { canvas, host, scope }
}

describe('terminalMenuHandleFor (production DOM shape)', () => {
  it('finds the handle registered on the xterm host, not the scope div', () => {
    const { canvas, host } = mountProductionShape()

    const registered = handle()
    const unregister = registerTerminalContextMenu(host, registered)

    expect(terminalMenuHandleFor(canvas)).toBe(registered)
    unregister()
  })

  it('returns null outside any terminal scope', () => {
    const outside = document.createElement('div')
    document.body.appendChild(outside)

    expect(terminalMenuHandleFor(outside)).toBeNull()
  })
})

describe('commandFocusedTerminal', () => {
  it('runs reload on the terminal holding DOM focus', () => {
    const { host, scope } = mountProductionShape()
    const textarea = document.createElement('textarea')
    host.appendChild(textarea)
    textarea.focus()

    const reload = vi.fn()
    const unregister = registerTerminalContextMenu(host, handle({ reload }))

    expect(commandFocusedTerminal('reload')).toBe(true)
    expect(reload).toHaveBeenCalledTimes(1)
    unregister()
    scope.remove()
  })

  it('answers false when focus is outside every terminal', () => {
    const outside = document.createElement('div')
    document.body.appendChild(outside)
    outside.focus?.()

    expect(commandFocusedTerminal('reload')).toBe(false)
  })

  it('answers false for back and forward — no terminal verb', () => {
    const { host, scope } = mountProductionShape()
    const textarea = document.createElement('textarea')
    host.appendChild(textarea)
    textarea.focus()

    const reload = vi.fn()
    const unregister = registerTerminalContextMenu(host, handle({ reload }))

    expect(commandFocusedTerminal('back')).toBe(false)
    expect(commandFocusedTerminal('forward')).toBe(false)
    expect(reload).not.toHaveBeenCalled()
    unregister()
    scope.remove()
  })

  it('unregisters idempotently — a later handle for the same scope replaces, an old remove does not', () => {
    const { host, scope } = mountProductionShape()
    const textarea = document.createElement('textarea')
    host.appendChild(textarea)
    textarea.focus()

    const first = handle()
    const second = handle()
    const removeFirst = registerTerminalContextMenu(host, first)
    registerTerminalContextMenu(host, second)

    removeFirst()

    expect(commandFocusedTerminal('reload')).toBe(true)
    expect(first.reload).not.toHaveBeenCalled()
    expect(second.reload).toHaveBeenCalledTimes(1)
    scope.remove()
  })
})

/** Mirror the production DOM shape (instance.tsx): the handle is registered
 * on the inner xterm host while the markers live on the enclosing scope. */
function mountScope(markers: 'interactive' | 'readonly'): { host: HTMLElement; scope: HTMLElement } {
  const scope = document.createElement('div')
  scope.dataset.terminal = ''

  if (markers === 'interactive') {
    scope.dataset.interactiveTerminal = ''
  }

  const host = document.createElement('div')
  scope.append(host)
  document.body.append(scope)

  return { host, scope }
}

describe('wordEraseFocusedTerminal', () => {
  const cleanups: Array<() => void> = []

  afterEach(() => {
    for (const cleanup of cleanups.splice(0)) {
      cleanup()
    }

    document.body.replaceChildren()
  })

  it('routes the close chord to the focused interactive terminal and reports it consumed', () => {
    const { host, scope } = mountScope('interactive')
    const wordErase = vi.fn(() => true)
    cleanups.push(registerTerminalContextMenu(host, {
      getSelection: () => '',
      paste: () => undefined,
      reload: () => {},
      selectAll: () => undefined,
      wordErase
    }))

    scope.tabIndex = -1
    scope.focus()

    expect(wordEraseFocusedTerminal()).toBe(true)
    expect(wordErase).toHaveBeenCalledOnce()
  })

  it('keys the registry by the [data-terminal] scope, not the nested host', () => {
    const { host } = mountScope('interactive')
    const wordErase = vi.fn(() => true)
    cleanups.push(registerTerminalContextMenu(host, {
      getSelection: () => '',
      paste: () => undefined,
      reload: () => {},
      selectAll: () => undefined,
      wordErase
    }))

    // The resolver walks up from a focus point INSIDE the host, exactly like
    // the real xterm textareaFocus.
    const focusPoint = document.createElement('textarea')
    host.append(focusPoint)
    focusPoint.focus()

    expect(wordEraseFocusedTerminal()).toBe(true)
    expect(terminalMenuHandleFor(focusPoint)).not.toBeNull()
  })

  it('keeps the read-only agent mirror closeable (null wordErase)', () => {
    const { host, scope } = mountScope('readonly')
    cleanups.push(registerTerminalContextMenu(host, {
      getSelection: () => '',
      paste: null,
      reload: () => {},
      selectAll: () => undefined,
      wordErase: null
    }))

    scope.tabIndex = -1
    scope.focus()

    expect(wordEraseFocusedTerminal()).toBe(false)
  })

  it('reports not consumed when the terminal has no live session', () => {
    const { host, scope } = mountScope('interactive')
    cleanups.push(registerTerminalContextMenu(host, {
      getSelection: () => '',
      paste: () => undefined,
      reload: () => {},
      selectAll: () => undefined,
      wordErase: () => false
    }))

    scope.tabIndex = -1
    scope.focus()

    // A sessionless terminal falls back to closing the tab.
    expect(wordEraseFocusedTerminal()).toBe(false)
  })

  it('reports not consumed when focus is elsewhere', () => {
    const { host } = mountScope('interactive')
    cleanups.push(registerTerminalContextMenu(host, {
      getSelection: () => '',
      paste: () => undefined,
      reload: () => {},
      selectAll: () => undefined,
      wordErase: () => true
    }))

    const elsewhere = document.createElement('textarea')
    document.body.append(elsewhere)
    elsewhere.focus()

    expect(wordEraseFocusedTerminal()).toBe(false)
  })

  it('stops routing after the handle unregisters (idempotent remove)', () => {
    const { host, scope } = mountScope('interactive')
    const wordErase = vi.fn(() => true)
    const unregister = registerTerminalContextMenu(host, {
      getSelection: () => '',
      paste: () => undefined,
      reload: () => {},
      selectAll: () => undefined,
      wordErase
    })

    scope.tabIndex = -1
    scope.focus()
    unregister()

    expect(wordEraseFocusedTerminal()).toBe(false)
    expect(wordErase).not.toHaveBeenCalled()
  })
})
