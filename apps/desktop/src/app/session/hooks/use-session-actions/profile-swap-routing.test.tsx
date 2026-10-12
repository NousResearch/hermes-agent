// Regression (#81817 / #79406): during a pending gateway profile swap the
// draft's quick-create selection ($newChatProfile) is already cleared while
// the swap is still opening the TARGET profile's backend. The create used to
// fall back to the still-live PREVIOUS profile, binding the new chat to (and
// inheriting the cwd of) the profile the user just left.
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile, $gatewaySwapTarget, $newChatProfile } from '@/store/profile'
import { setSessions } from '@/store/session'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

type Handle = Pick<ReturnType<typeof useSessionActions>, 'createBackendSessionForSend'>

function Harness({
  onReady,
  requestGateway
}: {
  onReady: (handle: Handle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'route:swap',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady({ createBackendSessionForSend: actions.createBackendSessionForSend })
  }, [actions, onReady])

  return null
}

async function createWith(
  profileSetup: () => void,
  beforeCreate?: () => void
): Promise<Record<string, unknown> | undefined> {
  let createParams: Record<string, unknown> | undefined

  const requestGateway = vi.fn(async <T,>(method: string, params?: Record<string, unknown>): Promise<T> => {
    if (method === 'session.create') {
      createParams = params

      return { session_id: 'runtime-swap', stored_session_id: null } as T
    }

    return {} as T
  })

  profileSetup()

  let handle: Handle | undefined
  render(<Harness onReady={h => (handle = h)} requestGateway={requestGateway as never} />)
  await waitFor(() => expect(handle).toBeDefined())

  if (beforeCreate) {
    await act(async () => {
      beforeCreate()
    })
  }

  await act(async () => {
    await handle!.createBackendSessionForSend()
  })

  return createParams
}

describe('createBackendSessionForSend × pending gateway swap', () => {
  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $gatewaySwapTarget.set(null)
    $activeGatewayProfile.set('default')
    setSessions([])
    vi.restoreAllMocks()
  })

  it('routes a plain new chat to the pending swap target, not the profile being left', async () => {
    const params = await createWith(
      () => {
        $activeGatewayProfile.set('coder')
        $newChatProfile.set(null)
      },
      // The swap is in flight while the UI is already live: set the pending
      // target AFTER mount (a switch that finishes clears it in `finally`).
      () => {
        $gatewaySwapTarget.set('analyst')
      }
    )

    expect(params).toMatchObject({ profile: 'analyst' })
  })

  it('still routes to the live gateway profile when no swap is pending', async () => {
    const params = await createWith(() => {
      $activeGatewayProfile.set('coder')
      $newChatProfile.set(null)
    })

    expect(params).toMatchObject({ profile: 'coder' })
  })
})
