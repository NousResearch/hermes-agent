import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $currentModel,
  $currentProvider,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

let handleEvent: ((event: RpcEvent) => void) | null = null

function Harness({ activeSessionId }: { activeSessionId: string | null }) {
  const sessionIdRef = useRef<string | null>(activeSessionId)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  sessionIdRef.current = activeSessionId

  const stream = useMessageStream({
    activeSessionIdRef: sessionIdRef,
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

async function mountStream(activeSessionId: string | null) {
  render(<Harness activeSessionId={activeSessionId} />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

describe('session.info does not clobber composer model selection', () => {
  beforeEach(() => {
    handleEvent = null
    setCurrentModel('deepseek-v4-flash')
    setCurrentProvider('deepseek')
    setCurrentModelSource('manual')
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentModelSource('')
  })

  it('keeps a sticky manual pick when a global session.info carries the profile default', async () => {
    await mountStream(null)

    act(() =>
      handleEvent!({
        payload: { model: 'deepseek-chat', provider: 'deepseek' },
        type: 'session.info'
      })
    )

    expect($currentModel.get()).toBe('deepseek-v4-flash')
    expect($currentProvider.get()).toBe('deepseek')
  })

  it('keeps the composer pick when an unscoped session.info arrives with no live session', async () => {
    await mountStream(null)

    act(() =>
      handleEvent!({
        payload: { cwd: '/tmp/project', model: 'deepseek-chat', provider: 'deepseek' },
        type: 'session.info'
      })
    )

    expect($currentModel.get()).toBe('deepseek-v4-flash')
    expect($currentProvider.get()).toBe('deepseek')
  })
})
