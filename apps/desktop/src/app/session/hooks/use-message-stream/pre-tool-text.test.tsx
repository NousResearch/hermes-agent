import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { type MutableRefObject, useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStateRef: MutableRefObject<Map<string, ClientSessionState>> | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  // Expose for tests
  sessionStateRef = sessionStateByRuntimeIdRef

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)

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

function getState() {
  return sessionStateRef!.current.get(SID)!
}

describe('useMessageStream pre-tool-call text preservation', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStateRef = null
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('preserves text streamed before tool calls when the turn completes', async () => {
    await mountStream()

    // Simulate: message.start -> text delta (pre-tool narration) -> tool.start -> tool.complete -> text delta (post-tool) -> message.complete
    act(() => handleEvent!({ payload: {}, session_id: SID, type: 'message.start' }))

    // Pre-tool-call narration
    act(() => handleEvent!({ payload: { text: 'Here is what I found: ' }, session_id: SID, type: 'message.delta' }))
    act(() => handleEvent!({ payload: { text: 'let me check the files.' }, session_id: SID, type: 'message.delta' }))

    // Tool call
    act(() =>
      handleEvent!({
        payload: { name: 'search_files', tool_id: 'tool-1', args: { pattern: 'test' } },
        session_id: SID,
        type: 'tool.start'
      })
    )

    act(() =>
      handleEvent!({
        payload: { name: 'search_files', tool_id: 'tool-1', result: { matches: [] } },
        session_id: SID,
        type: 'tool.complete'
      })
    )

    // Post-tool-call narration
    act(() => handleEvent!({ payload: { text: ' The results show X.' }, session_id: SID, type: 'message.delta' }))

    // Completion — final_response only contains the last API call's text
    act(() =>
      handleEvent!({
        payload: { text: 'The results show X.' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const state = getState()
    const assistantMsg = state.messages.find(m => m.role === 'assistant')
    expect(assistantMsg).toBeDefined()

    const parts = assistantMsg!.parts

    // Should have: text (pre-tool), tool-call, text (post-tool)
    const textParts = parts.filter(p => p.type === 'text')
    const toolParts = parts.filter(p => p.type === 'tool-call')

    // Pre-tool-call narration must survive
    expect(textParts.length).toBeGreaterThanOrEqual(2)
    const allText = textParts.map(p => p.text).join('')
    expect(allText).toContain('Here is what I found:')
    expect(allText).toContain('let me check the files.')
    expect(allText).toContain('The results show X.')

    // Tool row must survive
    expect(toolParts.length).toBe(1)
    expect(toolParts[0].toolName).toBe('search_files')
  })
})
