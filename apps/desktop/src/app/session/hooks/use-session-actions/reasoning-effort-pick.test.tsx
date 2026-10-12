import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { pickCurrentReasoningEffort, setCurrentReasoningEffort } from '@/store/session'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

type Create = ReturnType<typeof useSessionActions>['createBackendSessionForSend']

function Harness({ onReady, requestGateway }: { onReady: (create: Create) => void; requestGateway: never }) {
  const ref = <T,>(current: T) => ({ current })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => onReady(actions.createBackendSessionForSend), [actions.createBackendSessionForSend, onReady])

  return null
}

async function createParams(): Promise<Record<string, unknown> | undefined> {
  let params: Record<string, unknown> | undefined

  const requestGateway = vi.fn(async (method: string, next?: Record<string, unknown>) => {
    if (method === 'session.create') {
      params = next
    }

    return { session_id: 'rt-new', stored_session_id: null }
  })

  let create: Create | null = null
  render(<Harness onReady={ready => (create = ready)} requestGateway={requestGateway as never} />)
  await waitFor(() => expect(create).not.toBeNull())
  await act(async () => {
    await create!()
  })

  return params
}

// #87036: a per-session reasoning_effort outranks per-model reasoning_overrides,
// so only a level the user picked for the draft may ride session.create.
describe('session.create reasoning effort', () => {
  afterEach(() => {
    cleanup()
    setCurrentReasoningEffort('')
  })

  it('sends a level picked in the composer', async () => {
    pickCurrentReasoningEffort('high')

    expect(await createParams()).toMatchObject({ reasoning_effort: 'high' })
  })

  it('omits a level seeded from config or restored from a session', async () => {
    pickCurrentReasoningEffort('high')
    setCurrentReasoningEffort('xhigh')

    expect(await createParams()).not.toHaveProperty('reasoning_effort')
  })
})
