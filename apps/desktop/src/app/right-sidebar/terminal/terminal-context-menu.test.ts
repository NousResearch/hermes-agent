import { afterEach, describe, expect, it, vi } from 'vitest'

import { commandFocusedTerminal, registerTerminalContextMenu, terminalMenuHandleFor } from './terminal-context-menu'

const handle = (overrides: Partial<Parameters<typeof registerTerminalContextMenu>[1]> = {}) => ({
  getSelection: () => '',
  paste: null,
  reload: vi.fn(),
  selectAll: vi.fn(),
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
