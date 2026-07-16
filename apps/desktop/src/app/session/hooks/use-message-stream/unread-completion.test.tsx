import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { AppView } from '@/app/routes'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $unreadFinishedSessionIds } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const ACTIVE_RUNTIME_ID = 'runtime-active'
const ACTIVE_STORED_ID = 'stored-active'
const BACKGROUND_RUNTIME_ID = 'runtime-background'
const BACKGROUND_STORED_ID = 'stored-background'

let focused = true
let hidden = false
let currentView: AppView = 'chat'
let handleEvent: ((event: RpcEvent) => void) | null = null

function storedIdForRuntime(sessionId: string): null | string {
  if (sessionId === ACTIVE_RUNTIME_ID) {
    return ACTIVE_STORED_ID
  }

  if (sessionId === BACKGROUND_RUNTIME_ID) {
    return BACKGROUND_STORED_ID
  }

  return null
}

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_RUNTIME_ID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    currentView,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current =
        sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedIdForRuntime(sessionId))

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

function complete(sessionId: string) {
  act(() => handleEvent!({ payload: { text: 'done' }, session_id: sessionId, type: 'message.complete' }))
}

function fail(sessionId: string) {
  act(() => handleEvent!({ payload: { message: 'boom' }, session_id: sessionId, type: 'error' }))
}

describe('useMessageStream unread completion tracking', () => {
  beforeEach(() => {
    focused = true
    hidden = false
    currentView = 'chat'
    handleEvent = null
    $unreadFinishedSessionIds.set([])
    $threadScrolledUp.set(false)
    vi.spyOn(globalThis.document, 'hasFocus').mockImplementation(() => focused)
    vi.spyOn(globalThis.document, 'hidden', 'get').mockImplementation(() => hidden)
  })

  afterEach(() => {
    cleanup()
    $unreadFinishedSessionIds.set([])
    $threadScrolledUp.set(false)
    vi.restoreAllMocks()
  })

  it('does not mark a completion unread while the user is watching that session', async () => {
    await mountStream()

    complete(ACTIVE_RUNTIME_ID)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('marks a background session completion unread by stored id', async () => {
    await mountStream()

    complete(BACKGROUND_RUNTIME_ID)

    expect($unreadFinishedSessionIds.get()).toEqual([BACKGROUND_STORED_ID])
  })

  it('marks the active session unread when it completes while the window is unfocused', async () => {
    focused = false
    await mountStream()

    complete(ACTIVE_RUNTIME_ID)

    expect($unreadFinishedSessionIds.get()).toEqual([ACTIVE_STORED_ID])
  })

  it('marks the active session unread while the document is hidden despite stale focus state', async () => {
    hidden = true
    await mountStream()

    complete(ACTIVE_RUNTIME_ID)

    expect($unreadFinishedSessionIds.get()).toEqual([ACTIVE_STORED_ID])
  })

  it('keeps the active completion unread while the transcript is scrolled up', async () => {
    $threadScrolledUp.set(true)
    await mountStream()

    complete(ACTIVE_RUNTIME_ID)

    expect($unreadFinishedSessionIds.get()).toEqual([ACTIVE_STORED_ID])
  })

  it('marks the active session unread while another app view covers its transcript', async () => {
    currentView = 'settings'
    await mountStream()

    complete(ACTIVE_RUNTIME_ID)

    expect($unreadFinishedSessionIds.get()).toEqual([ACTIVE_STORED_ID])
  })

  it('also marks an off-screen failed turn unread', async () => {
    await mountStream()

    fail(BACKGROUND_RUNTIME_ID)

    expect($unreadFinishedSessionIds.get()).toEqual([BACKGROUND_STORED_ID])
  })
})
