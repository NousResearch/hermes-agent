import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessions, $unreadFinishedSessionIds, $workingSessionIds } from '@/store/session'
import { $sessionActivityIds } from '@/store/session-activity'
import { $subagentsBySession } from '@/store/subagents'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const RUNTIME_ID = 'runtime-parent'
const STORED_ID = 'stored-parent'
const LINEAGE_ROOT_ID = 'lineage-root'
const CHILD_ID = 'stored-review'

let handleEvent: ((event: RpcEvent) => void) | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(RUNTIME_ID)

  const sessionStateByRuntimeIdRef = useRef(
    new Map<string, ClientSessionState>([[RUNTIME_ID, createClientSessionState(STORED_ID)]])
  )

  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    currentView: 'chat',
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

const subagentEvent = (status: 'completed' | 'running'): RpcEvent => ({
  payload: {
    child_session_id: CHILD_ID,
    goal: 'Independent review',
    status,
    subagent_id: 'review-1'
  },
  session_id: RUNTIME_ID,
  type: status === 'running' ? 'subagent.start' : 'subagent.complete'
})

describe('useMessageStream subagent session activity', () => {
  beforeEach(() => {
    handleEvent = null
    $sessions.set([{ _lineage_root_id: LINEAGE_ROOT_ID, id: STORED_ID } as never])
    $workingSessionIds.set([])
    $subagentsBySession.set({})
    $unreadFinishedSessionIds.set([])
  })

  afterEach(() => {
    cleanup()
    $sessions.set([])
    $workingSessionIds.set([])
    $subagentsBySession.set({})
    $unreadFinishedSessionIds.set([])
    vi.restoreAllMocks()
  })

  it('keeps both the parent row and independent-review row active after the parent turn settles', async () => {
    await mountStream()

    act(() => handleEvent!(subagentEvent('running')))

    expect($subagentsBySession.get()[RUNTIME_ID]?.[0]?.ownerSessionId).toBe(LINEAGE_ROOT_ID)
    expect($sessionActivityIds.get()).toEqual([LINEAGE_ROOT_ID, CHILD_ID, STORED_ID])

    act(() =>
      handleEvent!({
        payload: { text: 'The review continues in the background.' },
        session_id: RUNTIME_ID,
        type: 'message.complete'
      })
    )

    expect($workingSessionIds.get()).toEqual([])
    expect($sessionActivityIds.get()).toEqual([LINEAGE_ROOT_ID, CHILD_ID, STORED_ID])
    expect($subagentsBySession.get()[RUNTIME_ID]?.[0]?.detached).toBe(true)
    $unreadFinishedSessionIds.set([])

    act(() =>
      handleEvent!({
        payload: {},
        session_id: RUNTIME_ID,
        type: 'message.start'
      })
    )

    expect($sessionActivityIds.get()).toEqual([LINEAGE_ROOT_ID, CHILD_ID, STORED_ID])

    act(() => {
      $sessions.set([{ _lineage_root_id: LINEAGE_ROOT_ID, id: 'second-continuation' } as never])
    })

    expect($sessionActivityIds.get()).toEqual([LINEAGE_ROOT_ID, CHILD_ID, 'second-continuation'])

    act(() => handleEvent!(subagentEvent('completed')))

    expect($subagentsBySession.get()[RUNTIME_ID]?.[0]?.handoff).toBe(true)
    expect($sessionActivityIds.get()).toEqual([LINEAGE_ROOT_ID, CHILD_ID, 'second-continuation'])
    expect($unreadFinishedSessionIds.get()).toEqual([LINEAGE_ROOT_ID])

    act(() =>
      handleEvent!({
        payload: {},
        session_id: RUNTIME_ID,
        type: 'message.start'
      })
    )

    expect($sessionActivityIds.get()).toEqual([])
  })
})
