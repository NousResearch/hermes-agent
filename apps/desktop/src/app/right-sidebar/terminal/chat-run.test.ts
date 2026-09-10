import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $chatTerminalRunRequest, $terminalInjection, $terminalTakeover } from '@/app/right-sidebar/store'
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
  handoffChatTerminalRunRequest,
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
    $terminalInjection.set(null)
    $terminalTakeover.set(false)
    $currentCwd.set('/workspace')
    installTerminalBridge()
  })

  afterEach(() => {
    $chatTerminalRunRequest.set(null)
    $terminalInjection.set(null)
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

  it('hands a queued request to the already-live exact terminal synchronously', () => {
    const terminalId = queueChatCommandInFreshTerminal('echo queued')!
    const seen: string[] = []
    const unsubscribe = $terminalInjection.subscribe(command => {
      if (!command) {
        return
      }

      expect($chatTerminalRunRequest.get()).toBeNull()
      seen.push(command)
      $terminalInjection.set(null)
    })

    expect(handoffChatTerminalRunRequest(terminalId)).toBe(true)
    expect(seen).toEqual(['echo queued'])
    expect($chatTerminalRunRequest.get()).toBeNull()
    expect($terminalInjection.get()).toBeNull()

    unsubscribe()
  })

  it('binds handoff to the active target id and fails closed without a live injection consumer', () => {
    const targetId = queueChatCommandInFreshTerminal('printf "hello"')!
    const otherId = createTerminal('/other')

    expect($activeTerminalId.get()).toBe(otherId)
    expect(handoffChatTerminalRunRequest(targetId)).toBe(false)
    expect($chatTerminalRunRequest.get()?.terminalId).toBe(targetId)

    $activeTerminalId.set(targetId)
    expect(handoffChatTerminalRunRequest(targetId)).toBe(false)
    expect($chatTerminalRunRequest.get()).toBeNull()
    expect($terminalInjection.get()).toBeNull()
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
