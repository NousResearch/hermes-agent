import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClientSessionState } from '@/app/types'
import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'
import { useMessageStream } from './index'

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
// Shared reference so tests can read/write session state directly.
const sessionRef: { current: Map<string, ClientSessionState> } = { current: new Map() }

function Harness() {
  const activeSessionIdRef = useRef(SID)
  const queryClientRef = useRef(new QueryClient())
  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef: sessionRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionRef.current.set(sessionId, next)
      return next
    }
  })
  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])
  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

function getMessages(): ChatMessage[] {
  return sessionRef.current.get(SID)?.messages ?? []
}

describe('useMessageStream multi-round text preservation', () => {
  beforeEach(() => {
    handleEvent = null
    sessionRef.current = new Map()
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('preserves tool + text parts when streamedText endsWith visibleFinalText', async () => {
    await mountStream()

    const msgId = 'assistant-stream-1'
    const parts: ChatMessagePart[] = [
      { type: 'text', text: 'Let me check the code.' },
      {
        type: 'tool-call',
        toolCallId: 'tool-1',
        toolName: 'search',
        args: { q: 'test' } as never,
        argsText: JSON.stringify({ q: 'test' }),
        result: { summary: 'results' } as never,
        isError: false
      },
      { type: 'text', text: 'The fix looks good.' }
    ]

    // Seed session state with a partially-streamed assistant message containing
    // both text and tool parts — simulates the text → tool → text sequence.
    sessionRef.current.set(SID, {
      ...createClientSessionState(),
      streamId: msgId,
      sawAssistantPayload: true,
      messages: [{ id: msgId, role: 'assistant', parts, pending: true }]
    })

    // message.complete final text matches the suffix of accumulated streamed text
    act(() =>
      handleEvent!({
        payload: { text: 'The fix looks good.' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const messages = getMessages()
    expect(messages).toHaveLength(1)
    expect(messages[0].pending).toBe(false)

    // Guard triggered — all parts returned as-is, tool-call preserved
    expect(messages[0].parts).toEqual(parts)

    const toolParts = messages[0].parts.filter(p => p.type === 'tool-call')
    expect(toolParts).toHaveLength(1)
    expect(toolParts[0].toolName).toBe('search')
  })

  it('replaces text but preserves non-text parts when final text is different', async () => {
    await mountStream()

    const msgId = 'assistant-stream-2'
    const parts: ChatMessagePart[] = [
      { type: 'text', text: 'Let me check the code.' },
      {
        type: 'tool-call',
        toolCallId: 'tool-2',
        toolName: 'search',
        args: { q: 'test' } as never,
        argsText: JSON.stringify({ q: 'test' }),
        result: { summary: 'results' } as never,
        isError: false
      }
    ]

    sessionRef.current.set(SID, {
      ...createClientSessionState(),
      streamId: msgId,
      sawAssistantPayload: true,
      messages: [{ id: msgId, role: 'assistant', parts, pending: true }]
    })

    // Final text bears no relation to streamed text — guard does NOT fire
    act(() =>
      handleEvent!({
        payload: { text: 'Something completely different.' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const messages = getMessages()
    expect(messages).toHaveLength(1)

    // Old text parts stripped, new final text present
    const textParts = messages[0].parts.filter(p => p.type === 'text')
    expect(textParts).toHaveLength(1)
    expect((textParts[0] as { text: string }).text).toContain('Something completely different.')

    // Tool-call part survives (replaceTextPart only removes text/reasoning parts)
    const toolParts = messages[0].parts.filter(p => p.type === 'tool-call')
    expect(toolParts).toHaveLength(1)
  })

  it('preserves multi-round tool calls when streamed suffix matches final text exactly', async () => {
    await mountStream()

    const msgId = 'assistant-stream-3'
    const parts: ChatMessagePart[] = [
      { type: 'text', text: 'First observation.' },
      {
        type: 'tool-call',
        toolCallId: 'tool-3a',
        toolName: 'read',
        args: { path: '/a' } as never,
        argsText: JSON.stringify({ path: '/a' }),
        result: { content: 'data' } as never,
        isError: false
      },
      { type: 'text', text: 'More analysis.' },
      {
        type: 'tool-call',
        toolCallId: 'tool-3b',
        toolName: 'grep',
        args: { pattern: 'foo' } as never,
        argsText: JSON.stringify({ pattern: 'foo' }),
        result: { lines: [] } as never,
        isError: false
      },
      { type: 'text', text: 'Conclusion reached.' }
    ]

    sessionRef.current.set(SID, {
      ...createClientSessionState(),
      streamId: msgId,
      sawAssistantPayload: true,
      messages: [{ id: msgId, role: 'assistant', parts, pending: true }]
    })

    act(() =>
      handleEvent!({
        payload: { text: 'Conclusion reached.' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const messages = getMessages()
    expect(messages[0].parts.filter(p => p.type === 'tool-call')).toHaveLength(2)
    expect(messages[0].parts).toEqual(parts)
  })
})
