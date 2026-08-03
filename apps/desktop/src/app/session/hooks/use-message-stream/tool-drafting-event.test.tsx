import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $draftingToolSessions } from '@/store/tool-drafting'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
const OTHER_SID = 'session-2'

// Module-scoped so a test can seed session state (e.g. interrupted) before the
// handler reads it — `sessionInterrupted` resolves against this map.
const sessionStates = new Map<string, ClientSessionState>()
let handleEvent: ((event: RpcEvent) => void) | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(sessionStates)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const next = updater(sessionStates.get(sessionId) ?? createClientSessionState())
      sessionStates.set(sessionId, next)

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

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}, sessionId = SID) {
  act(() => handleEvent!({ payload, session_id: sessionId, type }))
}

function draftedTool(sessionId = SID) {
  return $draftingToolSessions.get()[sessionId]?.name
}

describe('drafting-tool label lifecycle', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStates.clear()
    $draftingToolSessions.set({})
  })

  afterEach(() => {
    cleanup()
    sessionStates.clear()
    $draftingToolSessions.set({})
    vi.restoreAllMocks()
  })

  it('names the tool the model is drafting', async () => {
    await mountStream()

    emit('tool.generating', { name: 'write_file' })

    expect(draftedTool()).toBe('write_file')
  })

  // The label used to be retired only by the events that mean "this tool ran".
  // A tool can be abandoned without ever reaching `tool.start` — a mid-stream
  // retry drops a partial call, a guardrail-blocked tool skips the lifecycle
  // callbacks — and the name then sat on screen for the rest of the turn.
  it.each([
    ['message.delta', { text: 'never mind' }],
    ['reasoning.delta', { text: 'reconsidering' }],
    ['thinking.delta', { text: 'reconsidering' }],
    ['tool.start', { name: 'terminal', tool_id: 'tool-1' }],
    ['tool.complete', { name: 'terminal', tool_id: 'tool-1' }],
    ['message.complete', { text: 'done' }],
    ['error', { message: 'boom' }]
  ] as const)('retires the label when %s proves the model moved on', async (type, payload) => {
    await mountStream()
    emit('tool.generating', { name: 'write_file' })

    emit(type, payload)

    expect(draftedTool()).toBeUndefined()
  })

  it('leaves another session’s label alone', async () => {
    await mountStream()
    emit('tool.generating', { name: 'patch' }, OTHER_SID)
    emit('tool.generating', { name: 'write_file' })

    emit('message.delta', { text: 'moving on' })

    expect(draftedTool()).toBeUndefined()
    expect(draftedTool(OTHER_SID)).toBe('patch')
  })

  // A stopped turn can still emit a frame or two before the backend notices.
  it('ignores a tool announced after the user hit stop', async () => {
    sessionStates.set(SID, { ...createClientSessionState(), interrupted: true })
    await mountStream()

    emit('tool.generating', { name: 'write_file' })

    expect(draftedTool()).toBeUndefined()
  })
})
