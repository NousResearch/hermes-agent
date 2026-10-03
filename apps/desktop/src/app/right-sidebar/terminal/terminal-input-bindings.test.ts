import type { Terminal } from '@xterm/xterm'
import { expect, it, vi } from 'vitest'

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
