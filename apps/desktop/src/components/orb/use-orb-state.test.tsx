// useOrbState — wiring test: the hook projects the session stores onto
// `OrbState` for the thread dot and the composer jewel. (The precedence
// contract itself lives in orb-state.test.ts, against `resolveOrbState`.)

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage, type ChatMessagePart } from '@/lib/chat-messages'
import { $clarifyRequests } from '@/store/clarify'
import { $activeSessionId, $busy, $messages, $selectedStoredSessionId } from '@/store/session'

import { useOrbState } from './use-orb-state'

function assistantMessage(id: string, parts: ChatMessagePart[], extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, parts, role: 'assistant', ...extra }
}

function toolCallPart(result?: unknown): ChatMessagePart {
  return {
    args: {},
    argsText: '{}',
    toolCallId: 'call-1',
    toolName: 'shell',
    type: 'tool-call',
    ...(result !== undefined ? { result } : {})
  } as ChatMessagePart
}

describe('useOrbState', () => {
  afterEach(() => {
    cleanup()
    $messages.set([])
    $busy.set(false)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $clarifyRequests.set({})
    vi.useRealTimers()
  })

  it('is idle with no activity', () => {
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('idle')
  })

  it('is thinking while busy with no message parts yet', () => {
    $busy.set(true)
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('thinking')
  })

  it('is streaming while the tail assistant message has text', () => {
    $busy.set(true)
    $messages.set([assistantMessage('a1', [assistantTextPart('hello')], { pending: true })])
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('streaming')
  })

  it('is tool-running while a tool call is in flight', () => {
    $busy.set(true)
    $messages.set([assistantMessage('a1', [toolCallPart()], { pending: true })])
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('tool-running')
  })

  it('is tool-result once a tool result lands', () => {
    $busy.set(true)
    $messages.set([assistantMessage('a1', [toolCallPart({ ok: true })], { pending: true })])
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('tool-result')
  })

  it('is error when the tail assistant message failed', () => {
    $messages.set([assistantMessage('a1', [assistantTextPart('x')], { error: 'boom' })])
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('error')
  })

  it('is waiting-input when the turn is parked on a prompt', () => {
    $busy.set(true)
    $clarifyRequests.set({
      '': { choices: null, multiSelect: false, question: 'Pick one', requestId: 'r1', sessionId: null }
    })
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('waiting-input')
  })

  it('flashes complete after a successful turn settles, then idles', () => {
    vi.useFakeTimers()
    $busy.set(true)
    const { result } = renderHook(() => useOrbState())

    expect(result.current).toBe('thinking')

    act(() => {
      $busy.set(false)
    })
    expect(result.current).toBe('complete')

    act(() => {
      vi.advanceTimersByTime(2600)
    })
    expect(result.current).toBe('idle')
  })
})
