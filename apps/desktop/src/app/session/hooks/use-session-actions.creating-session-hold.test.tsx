import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import { setCurrentCwd, setNewChatWorkspaceTarget } from '@/store/session'

import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => {
  const original = await importOriginal<Record<string, unknown>>()

  return {
    ...original,
    // Default-preserving spy: tests that route by the active source override it.
    activeGatewayConnectionId: vi.fn(original.activeGatewayConnectionId as () => null | string),
    requestGatewayForAgent: vi.fn(),
    requestGatewayForProfile: vi.fn(),
    retainGatewayForAgent: vi.fn(async () => () => undefined)
  }
})

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

describe('createBackendSessionForSend creatingSessionRef hold (#66057)', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $newChatProfile.set(null)
    $activeGatewayProfile.set('default')
    setCurrentCwd('')
    setNewChatWorkspaceTarget(undefined)
    vi.restoreAllMocks()
  })

  function GuardHarness({
    creatingSessionRef,
    navigate,
    onReady,
    requestGateway,
    routeId,
    selectedStoredSessionIdRef
  }: {
    creatingSessionRef: MutableRefObject<boolean>
    navigate: (...args: never[]) => unknown
    onReady: (create: () => Promise<string | null>) => void
    requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
    routeId: null | string
    selectedStoredSessionIdRef: MutableRefObject<null | string>
  }) {
    const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

    const actions = useSessionActions({
      activeSessionId: null,
      activeSessionIdRef: ref<string | null>(null),
      busyRef: ref(false),
      creatingSessionRef,
      ensureSessionState: () => ({}) as ClientSessionState,
      getRouteToken: () => 'token',
      getRoutedStoredSessionId: () => routeId,
      navigate: navigate as never,
      requestGateway,
      resetViewSync: vi.fn(),
      routedSessionId: routeId,
      runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
      selectedStoredSessionId: selectedStoredSessionIdRef.current,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
      syncSessionStateToView: vi.fn(),
      updateSessionState: () => ({}) as ClientSessionState
    })

    useEffect(() => {
      onReady(() => actions.createBackendSessionForSend())
    }, [actions, onReady])

    return null
  }

  it('keeps creatingSessionRef true until routedSessionId catches up to the created stored id', async () => {
    const creatingSessionRef: MutableRefObject<boolean> = { current: false }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }
    const navigate = vi.fn()
    let routedSessionId: null | string = 'session-A'

    const requestGateway = async <T,>(method: string): Promise<T> => {
      if (method === 'session.create') {
        return { session_id: 'rt-new', stored_session_id: 'stored-new' } as T
      }

      return {} as T
    }

    let create: (() => Promise<string | null>) | null = null

    const { rerender } = render(
      <GuardHarness
        creatingSessionRef={creatingSessionRef}
        navigate={navigate}
        onReady={fn => (create = fn)}
        requestGateway={requestGateway}
        routeId={routedSessionId}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    await waitFor(() => expect(create).not.toBeNull())

    await act(async () => {
      await create!()
    })

    expect(navigate).toHaveBeenCalled()
    expect(creatingSessionRef.current).toBe(true)
    expect(selectedStoredSessionIdRef.current).toBe('stored-new')

    // Route still stale on A — guard must stay up (not a user navigation away).
    rerender(
      <GuardHarness
        creatingSessionRef={creatingSessionRef}
        navigate={navigate}
        onReady={() => undefined}
        requestGateway={requestGateway}
        routeId="session-A"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )
    expect(creatingSessionRef.current).toBe(true)

    // Router catches up to the created stored id — release the guard.
    routedSessionId = 'stored-new'
    rerender(
      <GuardHarness
        creatingSessionRef={creatingSessionRef}
        navigate={navigate}
        onReady={() => undefined}
        requestGateway={requestGateway}
        routeId={routedSessionId}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )
    expect(creatingSessionRef.current).toBe(false)
  })

  it('clears creatingSessionRef when navigate throws', async () => {
    const creatingSessionRef: MutableRefObject<boolean> = { current: false }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }

    const navigate = vi.fn(() => {
      throw new Error('navigate failed')
    })

    let create: (() => Promise<string | null>) | null = null
    render(
      <GuardHarness
        creatingSessionRef={creatingSessionRef}
        navigate={navigate}
        onReady={fn => (create = fn)}
        requestGateway={async method => {
          if (method === 'session.create') {
            return { session_id: 'rt-new', stored_session_id: 'stored-new' } as never
          }

          return {} as never
        }}
        routeId="session-A"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )
    await waitFor(() => expect(create).not.toBeNull())

    await act(async () => {
      await create!()
    })

    expect(navigate).toHaveBeenCalled()
    expect(creatingSessionRef.current).toBe(false)
  })

  it('clears creatingSessionRef when the route moves to a different session than pending', async () => {
    const creatingSessionRef: MutableRefObject<boolean> = { current: false }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }
    const navigate = vi.fn()

    const requestGateway = async <T,>(method: string): Promise<T> => {
      if (method === 'session.create') {
        return { session_id: 'rt-new', stored_session_id: 'stored-new' } as T
      }

      return {} as T
    }

    let create: (() => Promise<string | null>) | null = null

    const { rerender } = render(
      <GuardHarness
        creatingSessionRef={creatingSessionRef}
        navigate={navigate}
        onReady={fn => (create = fn)}
        requestGateway={requestGateway}
        routeId="session-A"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    await waitFor(() => expect(create).not.toBeNull())

    await act(async () => {
      await create!()
    })
    expect(creatingSessionRef.current).toBe(true)

    // User clicked another session while create navigate was still pending.
    rerender(
      <GuardHarness
        creatingSessionRef={creatingSessionRef}
        navigate={navigate}
        onReady={() => undefined}
        requestGateway={requestGateway}
        routeId="session-C"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )
    expect(creatingSessionRef.current).toBe(false)
  })

  it('clears creatingSessionRef via safety timeout if the route never catches up', async () => {
    const creatingSessionRef: MutableRefObject<boolean> = { current: false }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }
    const navigate = vi.fn()

    let create: (() => Promise<string | null>) | null = null
    render(
      <GuardHarness
        creatingSessionRef={creatingSessionRef}
        navigate={navigate}
        onReady={fn => (create = fn)}
        requestGateway={async method => {
          if (method === 'session.create') {
            return { session_id: 'rt-new', stored_session_id: 'stored-new' } as never
          }

          return {} as never
        }}
        routeId="session-A"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )
    await waitFor(() => expect(create).not.toBeNull())

    // Arm the pending timeout under fake timers so we can advance deterministically.
    vi.useFakeTimers()
    await act(async () => {
      await create!()
    })
    expect(creatingSessionRef.current).toBe(true)
    expect(navigate).toHaveBeenCalledTimes(1)

    // Route stays on A forever — safety timeout must retry navigate (reconcile)
    // and drop the guard so use-route-resume can self-heal if the route still
    // never moves.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3_000)
    })
    expect(creatingSessionRef.current).toBe(false)
    expect(navigate).toHaveBeenCalledTimes(2)
    expect(navigate).toHaveBeenLastCalledWith('/stored-new', { replace: true })
  })
})
