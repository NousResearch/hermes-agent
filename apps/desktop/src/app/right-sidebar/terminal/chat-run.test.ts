import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $chatTerminalRunRequest, $terminalTakeover } from '@/app/right-sidebar/store'
import {
  $activeTerminalId,
  $terminals,
  closeAllTerminals,
  closeOtherTerminals,
  closeTerminal,
  createTerminal
} from '@/app/right-sidebar/terminal/terminals'
import { $currentCwd } from '@/store/session'

import {
  deliverChatTerminalRunRequest,
  hasEmbeddedTerminalBridge,
  isRunnableChatTerminalCommandText,
  MAX_CHAT_RUN_CHARS,
  queueChatCommandInFreshTerminal
} from './chat-run'

function installTerminalBridge() {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { terminal: { start: vi.fn(), write: vi.fn() } }
  })
}

describe('chat terminal Run queue', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $terminals.set([])
    $activeTerminalId.set(null)
    $chatTerminalRunRequest.set(null)
    $terminalTakeover.set(false)
    $currentCwd.set('/workspace')
    installTerminalBridge()
  })

  afterEach(() => {
    $chatTerminalRunRequest.set(null)
    $terminals.set([])
    $activeTerminalId.set(null)
    $terminalTakeover.set(false)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    window.localStorage.clear()
  })

  it('rejects hidden terminal-control and bidi payloads while allowing visible multiline shell text', () => {
    for (const unsafe of [
      'echo ok\rwhoami',
      'echo ok\x1b[2J',
      'echo ok\0id',
      `echo safe\u202E ; rm -rf /`,
      `echo safe\u2066id\u2069`,
      `echo safe\u200b && id`,
      `echo safe\u034fid`,
      `echo safe\u2061id`,
      `echo safe\u2062id`,
      `echo safe${String.fromCodePoint(0xe0069)}id`,
      `echo safe${String.fromCodePoint(0xe0100)}id`
    ]) {
      expect(isRunnableChatTerminalCommandText(unsafe)).toBe(false)
    }

    expect(isRunnableChatTerminalCommandText('printf "one\\ntwo"\necho done')).toBe(true)
    expect(isRunnableChatTerminalCommandText('\t echo ok')).toBe(true)
    expect(isRunnableChatTerminalCommandText('x'.repeat(MAX_CHAT_RUN_CHARS))).toBe(true)
    expect(isRunnableChatTerminalCommandText('x'.repeat(MAX_CHAT_RUN_CHARS + 1))).toBe(false)
  })

  it('creates a fresh user terminal instead of injecting into the existing active shell', () => {
    const oldId = createTerminal('/old-shell')
    const newId = queueChatCommandInFreshTerminal('echo hello')

    expect(newId).toBeTruthy()
    expect(newId).not.toBe(oldId)
    expect($activeTerminalId.get()).toBe(newId)
    expect($terminals.get().find(term => term.id === newId)).toMatchObject({ cwd: '/workspace', kind: 'user' })
    expect($chatTerminalRunRequest.get()).toEqual({ command: 'echo hello', terminalId: newId })
    expect($terminalTakeover.get()).toBe(true)
  })

  it('delivers a request queued before the terminal-session subscription attaches', () => {
    const terminalId = queueChatCommandInFreshTerminal('echo queued')!
    const write = vi.fn(async () => true)

    const unsubscribe = $chatTerminalRunRequest.subscribe(request => {
      if (request?.terminalId === terminalId) {
        deliverChatTerminalRunRequest(terminalId, 'pty-late', write)
      }
    })

    expect(write).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledWith('pty-late', 'echo queued\r')
    expect($chatTerminalRunRequest.get()).toBeNull()

    unsubscribe()
  })

  it('binds delivery to the target id, writes exact bytes + Enter, and clears before the bridge call', () => {
    const targetId = queueChatCommandInFreshTerminal('printf "hello"')!

    const write = vi.fn(async () => {
      expect($chatTerminalRunRequest.get()).toBeNull()

      return true
    })

    expect(deliverChatTerminalRunRequest('wrong-terminal', 'pty-1', write)).toBe(false)
    expect(write).not.toHaveBeenCalled()
    expect($chatTerminalRunRequest.get()?.terminalId).toBe(targetId)

    expect(deliverChatTerminalRunRequest(targetId, 'pty-1', write)).toBe(true)
    expect(write).toHaveBeenCalledWith('pty-1', 'printf "hello"\r')
    expect($chatTerminalRunRequest.get()).toBeNull()

    expect(deliverChatTerminalRunRequest(targetId, 'pty-1', write)).toBe(false)
    expect(write).toHaveBeenCalledTimes(1)
  })

  it('refuses a second unflushed request and does not create an extra terminal', () => {
    const first = queueChatCommandInFreshTerminal('echo one')
    const count = $terminals.get().length
    const second = queueChatCommandInFreshTerminal('echo two')

    expect(first).toBeTruthy()
    expect(second).toBeNull()
    expect($terminals.get()).toHaveLength(count)
    expect($chatTerminalRunRequest.get()?.command).toBe('echo one')
  })

  it('cancels authorization whenever any terminal-list mutation removes the target', () => {
    let targetId = queueChatCommandInFreshTerminal('echo close')!
    closeTerminal(targetId)
    expect($chatTerminalRunRequest.get()).toBeNull()

    targetId = queueChatCommandInFreshTerminal('echo close-all')!
    closeAllTerminals()
    expect($chatTerminalRunRequest.get()).toBeNull()

    const survivor = createTerminal('/survivor')
    targetId = queueChatCommandInFreshTerminal('echo close-other')!
    closeOtherTerminals(survivor)
    expect($terminals.get().some(term => term.id === targetId)).toBe(false)
    expect($chatTerminalRunRequest.get()).toBeNull()
  })

  it('fails closed when the real Electron terminal write bridge is unavailable', () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop

    expect(hasEmbeddedTerminalBridge()).toBe(false)
    expect(queueChatCommandInFreshTerminal('echo hello')).toBeNull()
    expect($terminals.get()).toHaveLength(0)
  })
})
