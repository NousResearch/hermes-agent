import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessionStates, publishSessionState } from '@/store/session-states'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

// `session.reclaimed`: the backend tore down a live session we're still
// holding (idle TTL, LRU cap, WS-orphan reap). Before this event the runtime id
// stayed cached until something failed against it, which read to the user as
// the session vanishing rather than being reclaimed.

const ACTIVE_SID = 'session-active'
const ACTIVE_PROFILE = 'compass'
let handleEvent: ((event: RpcEvent) => void) | null = null
let queryClient: QueryClient

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

  const stream = useMessageStream({
    activeGatewayProfile: ACTIVE_PROFILE,
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient,
    refreshHermesConfig: vi.fn<() => Promise<void>>(async () => undefined),
    refreshSessions: vi.fn<() => Promise<void>>(async () => undefined),
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

const reclaim = (sessionId: string, reason = 'ws_orphan_reap') =>
  act(() =>
    handleEvent!({
      payload: { reason, session_id: sessionId, stored_session_id: 'stored-1' },
      session_id: '',
      type: 'session.reclaimed'
    } as RpcEvent)
  )

beforeEach(() => {
  handleEvent = null
  queryClient = new QueryClient()
  $sessionStates.set({})
})

afterEach(() => {
  cleanup()
  $sessionStates.set({})
  vi.restoreAllMocks()
})

describe('session.reclaimed', () => {
  it('drops the cached state for the reclaimed runtime', async () => {
    await mountStream()
    publishSessionState('live-gone', createClientSessionState())
    expect($sessionStates.get()['live-gone']).toBeDefined()

    reclaim('live-gone')

    expect($sessionStates.get()['live-gone']).toBeUndefined()
  })

  it('leaves every other live session alone', async () => {
    await mountStream()
    publishSessionState('live-gone', createClientSessionState())
    publishSessionState('live-kept', createClientSessionState())

    reclaim('live-gone')

    // Both halves matter: the target went, the bystander stayed. Asserting
    // only the survivor would pass with no handler at all.
    expect($sessionStates.get()['live-gone']).toBeUndefined()
    expect($sessionStates.get()['live-kept']).toBeDefined()
  })

  it('ignores a payload with no runtime id instead of clearing everything', async () => {
    await mountStream()
    publishSessionState('live-a', createClientSessionState())
    publishSessionState('live-b', createClientSessionState())

    reclaim('')

    // A malformed/empty id must be a no-op, never a blanket wipe.
    expect(Object.keys($sessionStates.get()).sort()).toEqual(['live-a', 'live-b'])
  })

  it('drops the runtime regardless of which reclaim reason fired', async () => {
    for (const reason of ['idle_timeout', 'lru_evict', 'ws_orphan_reap']) {
      $sessionStates.set({})
      cleanup()
      handleEvent = null
      await mountStream()
      publishSessionState('live-gone', createClientSessionState())

      reclaim('live-gone', reason)

      expect($sessionStates.get()['live-gone'], reason).toBeUndefined()
    }
  })
})
