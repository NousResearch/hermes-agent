import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'timeline-session'

let handleEvent: ((event: RpcEvent) => void) | null = null
let states: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

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
      states.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

const event = (type: string, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => handleEvent?.({ payload: { ...payload, timestamp }, session_id: SID, type }))

describe('live transcript timeline events', () => {
  beforeEach(async () => {
    handleEvent = null
    states = new Map()
    render(<Harness />)
    await waitFor(() => expect(handleEvent).not.toBeNull())
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('records commentary, tool, resumed text, and turn-stop boundaries', () => {
    event('message.start', 100)
    event('message.delta', 101.125, { text: 'Let me inspect it.' })
    event('message.interim', 101.75, { already_streamed: true, text: 'Let me inspect it.' })
    event('tool.start', 102.25, { args: { path: 'README.md' }, name: 'read_file', tool_id: 'call-1' })
    event('tool.complete', 104.5, { name: 'read_file', result: { content: 'ok' }, tool_id: 'call-1' })
    event('message.delta', 105.625, { text: 'The file looks good.' })
    event('message.complete', 106.875, { text: 'The file looks good.' })

    const assistants = states.get(SID)?.messages.filter(message => message.role === 'assistant') ?? []

    expect(assistants).toHaveLength(2)
    expect([assistants[0].timestamp, assistants[0].completedAt]).toEqual([101.125, 101.75])
    expect(assistants[0].parts.map(part => [part.timestamp, part.completedAt])).toEqual([[101.125, 101.75]])

    expect([assistants[1].timestamp, assistants[1].completedAt]).toEqual([102.25, 106.875])
    expect(assistants[1].parts.map(part => part.type)).toEqual(['tool-call', 'text'])
    expect(assistants[1].parts.map(part => [part.timestamp, part.completedAt])).toEqual([
      [102.25, 104.5],
      [105.625, 106.875]
    ])
  })

  it('preserves cross-channel delta order inside one flush window', () => {
    event('message.start', 200)
    event('reasoning.delta', 201.125, { text: 'Thinking first.' })
    event('message.delta', 202.25, { text: 'Then speaking.' })
    event('tool.start', 203.5, { args: {}, name: 'terminal', tool_id: 'call-2' })

    const assistant = states.get(SID)?.messages.find(message => message.role === 'assistant')

    expect(assistant?.parts.map(part => part.type)).toEqual(['reasoning', 'text', 'tool-call'])
    expect(assistant?.parts.map(part => part.timestamp)).toEqual([201.125, 202.25, 203.5])
  })

  it('uses the gateway event time for an error boundary', () => {
    event('message.start', 300)
    event('error', 301.875, { error: 'provider failed' })

    const assistant = states.get(SID)?.messages.find(message => message.role === 'assistant')

    expect(assistant?.error).toBeTruthy()
    expect([assistant?.timestamp, assistant?.completedAt]).toEqual([301.875, 301.875])
  })

  it('uses the gateway event time for a review summary system row', () => {
    event('review.summary', 401.625, { text: 'Review saved.' })

    const system = states.get(SID)?.messages.find(message => message.role === 'system')

    expect(system?.timestamp).toBe(401.625)
    expect(system?.parts[0].timestamp).toBe(401.625)
  })

  it('uses session.info time when it is the only stop boundary', () => {
    event('message.start', 500)
    event('tool.start', 501, { args: {}, name: 'terminal', tool_id: 'call-3' })
    event('session.info', 502.75, { running: false })

    const assistant = states.get(SID)?.messages.find(message => message.role === 'assistant')

    expect(assistant?.completedAt).toBe(502.75)
    expect(assistant?.parts[0].completedAt).toBe(502.75)
  })
})
