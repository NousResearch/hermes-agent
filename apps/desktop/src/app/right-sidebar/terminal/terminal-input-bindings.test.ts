import type { Terminal } from '@xterm/xterm'
import { expect, it, vi } from 'vitest'

import { markKeybindHandled } from '@/lib/keybinds/handled-keys'

import { terminalMenuHandleFor } from './terminal-context-menu'
import { bindTerminalClipboard } from './terminal-input-bindings'

it('routes word erase only to a live shell and releases the context-menu handle on cleanup', () => {
  const host = document.createElement('div')
  host.setAttribute('data-terminal', '')

  const term = {
    attachCustomKeyEventHandler: vi.fn(),
    getSelection: () => '',
    selectAll: vi.fn()
  } as unknown as Terminal

  const markActivity = vi.fn()
  const writeInput = vi.fn()
  let hasSession = true
  const dispose = bindTerminalClipboard(term, host, markActivity, writeInput, () => hasSession)
  const handle = terminalMenuHandleFor(host)!

  expect(handle.wordErase?.()).toBe(true)
  expect(writeInput).toHaveBeenCalledExactlyOnceWith('\x17')
  expect(markActivity).toHaveBeenCalledOnce()

  hasSession = false
  expect(handle.wordErase?.()).toBe(false)
  expect(writeInput).toHaveBeenCalledTimes(1)

  hasSession = true
  expect(handle.wordErase?.()).toBe(true)
  expect(writeInput).toHaveBeenCalledTimes(2)
  dispose()
  expect(terminalMenuHandleFor(host)).toBeNull()
})

it('keeps a plain Alt chord Hermes ran off the PTY and leaves shared Ctrl keys and other cancels to the shell', () => {
  const attachCustomKeyEventHandler = vi.fn()
  const term = { attachCustomKeyEventHandler, getSelection: () => '' } as unknown as Terminal
  const dispose = bindTerminalClipboard(term, document.createElement('div'), vi.fn(), vi.fn(), () => true)
  const passToPty = attachCustomKeyEventHandler.mock.calls[0]![0] as (event: KeyboardEvent) => boolean

  const ran = (init: KeyboardEventInit) => {
    const event = new KeyboardEvent('keydown', { cancelable: true, ...init })
    event.preventDefault()
    markKeybindHandled(event)

    return event
  }

  expect(passToPty(ran({ altKey: true, key: 't' }))).toBe(false)
  // Ctrl+D is EOF in the shell even though it is also a Hermes chord.
  expect(passToPty(ran({ ctrlKey: true, key: 'd' }))).toBe(true)
  expect(passToPty(new KeyboardEvent('keydown', { altKey: true, key: 'b' }))).toBe(true)

  // Cancelled by someone else (a tooltip's Escape, a declined passthrough
  // chord): still the shell's.
  const cancelled = new KeyboardEvent('keydown', { altKey: true, cancelable: true, key: '9' })
  cancelled.preventDefault()
  expect(passToPty(cancelled)).toBe(true)
  dispose()
})
