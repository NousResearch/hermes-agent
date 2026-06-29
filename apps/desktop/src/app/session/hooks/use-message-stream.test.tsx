import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $currentCwd, setCurrentCwd } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './use-message-stream'

const projectStoreMocks = vi.hoisted(() => ({
  followActiveSessionCwd: vi.fn(async () => undefined),
  refreshProjectTree: vi.fn(async () => undefined),
  refreshProjects: vi.fn(async () => undefined)
}))

vi.mock('@/store/projects', async importOriginal => {
  const actual = await importOriginal()

  return {
    ...actual,
    ...projectStoreMocks
  }
})

interface HarnessProps {
  activeSessionIdRef: MutableRefObject<null | string>
  onReady: (handleGatewayEvent: (event: RpcEvent) => void) => void
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, unknown>>
}

function MessageStreamHarness({ activeSessionIdRef, onReady, sessionStateByRuntimeIdRef }: HarnessProps) {
  const { handleGatewayEvent } = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: async () => undefined,
    queryClient: new QueryClient(),
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined,
    sessionStateByRuntimeIdRef: sessionStateByRuntimeIdRef as MutableRefObject<Map<string, never>>,
    updateSessionState: (_sessionId, updater) => updater({ interrupted: false, messages: [] } as never)
  })

  useEffect(() => {
    onReady(handleGatewayEvent)
  }, [handleGatewayEvent, onReady])

  return null
}

describe('useMessageStream session.info cwd refresh', () => {
  beforeEach(() => {
    setCurrentCwd('/old-root')
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    setCurrentCwd('')
  })

  it('refreshes project caches on the first active-session cwd change without following scope', () => {
    const activeSessionIdRef: MutableRefObject<null | string> = { current: 'sid-1' }
    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, unknown>> = { current: new Map() }
    let handleGatewayEvent: null | ((event: RpcEvent) => void) = null

    render(
      <MessageStreamHarness
        activeSessionIdRef={activeSessionIdRef}
        onReady={handler => {
          handleGatewayEvent = handler
        }}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    act(() => {
      handleGatewayEvent?.({
        payload: { cwd: '/new-root' },
        session_id: 'sid-1',
        type: 'session.info'
      } as RpcEvent)
    })

    expect($currentCwd.get()).toBe('/new-root')
    expect(projectStoreMocks.refreshProjects).toHaveBeenCalledTimes(1)
    expect(projectStoreMocks.refreshProjectTree).toHaveBeenCalledTimes(1)
    expect(projectStoreMocks.followActiveSessionCwd).not.toHaveBeenCalled()
  })

  it('follows scope on subsequent cwd moves for the same active session', () => {
    const activeSessionIdRef: MutableRefObject<null | string> = { current: 'sid-1' }
    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, unknown>> = { current: new Map() }
    let handleGatewayEvent: null | ((event: RpcEvent) => void) = null

    render(
      <MessageStreamHarness
        activeSessionIdRef={activeSessionIdRef}
        onReady={handler => {
          handleGatewayEvent = handler
        }}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    act(() => {
      handleGatewayEvent?.({
        payload: { cwd: '/new-root' },
        session_id: 'sid-1',
        type: 'session.info'
      } as RpcEvent)
    })

    act(() => {
      handleGatewayEvent?.({
        payload: { cwd: '/newer-root' },
        session_id: 'sid-1',
        type: 'session.info'
      } as RpcEvent)
    })

    expect(projectStoreMocks.refreshProjects).toHaveBeenCalledTimes(1)
    expect(projectStoreMocks.refreshProjectTree).toHaveBeenCalledTimes(1)
    expect(projectStoreMocks.followActiveSessionCwd).toHaveBeenCalledTimes(1)
    expect(projectStoreMocks.followActiveSessionCwd).toHaveBeenCalledWith('/newer-root')
  })
})
