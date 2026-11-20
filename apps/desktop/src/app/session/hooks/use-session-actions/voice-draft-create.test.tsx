import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute } from '@/store/profile'
import { $freshDraftKey, setFreshDraftReady } from '@/store/session'

import { deferred } from '../../../../test/deferred'
import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => undefined)
}))

const RUNTIME_SESSION_ID = 'rt-voice-001'

type Create = ReturnType<typeof useSessionActions>['createBackendSessionForSend']
type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

function Harness({ onReady, requestGateway }: { onReady: (create: Create) => void; requestGateway: RequestGateway }) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => '/::',
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

  useEffect(() => {
    onReady(actions.createBackendSessionForSend)
  }, [actions.createBackendSessionForSend, onReady])

  return null
}

async function mount(requestGateway: RequestGateway): Promise<Create> {
  let create: Create | null = null
  render(<Harness onReady={value => (create = value)} requestGateway={requestGateway} />)
  await waitFor(() => expect(create).not.toBeNull())

  return create!
}

// A GPT-Live call opened on New Chat creates its own chat: the call binds to it only
// while that draft is still on screen and its owner has not moved.
describe('fresh voice chat creation', () => {
  afterEach(() => {
    cleanup()
    setFreshDraftReady(false)
    $newChatRoute.set(null)
    $newChatProfile.set(null)
    $activeGatewayProfile.set('default')
    vi.mocked(requestGatewayForAgent).mockReset()
  })

  it('does not hand off a fresh voice create after the ambient owner changes', async () => {
    $activeGatewayProfile.set('voice-original')
    setFreshDraftReady(true)
    const created = deferred<never>()
    const requestGateway = vi.fn(async (method: string) => (method === 'session.create' ? created.promise : ({} as never)))
    const onAssigned = vi.fn()
    const create = await mount(requestGateway)

    let pending!: Promise<null | string>
    await act(async () => {
      pending = create(null, { voiceDraft: { key: $freshDraftKey.get(), isCurrent: () => true, onAssigned } })
    })
    expect(requestGateway).toHaveBeenCalledWith('session.create', expect.objectContaining({ profile: 'voice-original' }))
    await act(async () => {
      $activeGatewayProfile.set('voice-other')
      created.resolve({ session_id: RUNTIME_SESSION_ID, stored_session_id: 'voice-stored' } as never)
      await expect(pending).resolves.toBeNull()
    })
    expect(onAssigned).not.toHaveBeenCalled()
    // Never send the orphan's runtime id to the new owner.
    expect(requestGateway.mock.calls.map(([method]) => method)).not.toContain('session.close')
  })

  it.each([true, false])('hands off the exact voice owner only while the draft still owns the create (%s)', async current => {
    $newChatRoute.set({ connectionId: 'voice-host', profile: 'voice-label', targetProfile: 'voice-backend' })
    $activeGatewayProfile.set('other-profile')
    setFreshDraftReady(true)
    const created = deferred<never>()
    const routed = vi.mocked(requestGatewayForAgent)
    routed.mockImplementation(async (_connection, _profile, method) => (method === 'session.create' ? created.promise : ({} as never)))
    const onAssigned = vi.fn()
    const isCurrent = vi.fn(() => true)
    const create = await mount(vi.fn())

    let pending!: Promise<null | string>
    await act(async () => {
      pending = create(null, { voiceDraft: { key: $freshDraftKey.get(), isCurrent, onAssigned } })
    })
    expect(routed).toHaveBeenCalledWith('voice-host', 'voice-label', 'session.create',
      expect.objectContaining({ profile: 'voice-backend' }), undefined, undefined, { spawnPriority: 'foreground' })
    expect(onAssigned).not.toHaveBeenCalled()
    // Text Send must not reuse the same create receipt while voice is in flight.
    await expect(create('typed concurrently')).resolves.toBeNull()
    isCurrent.mockReturnValue(current)
    await act(async () => {
      created.resolve({ session_id: RUNTIME_SESSION_ID, stored_session_id: 'voice-stored' } as never)
      await expect(pending).resolves.toBe(current ? RUNTIME_SESSION_ID : null)
    })

    if (current) {
      expect(onAssigned).toHaveBeenCalledExactlyOnceWith({
        backendProfile: 'voice-backend',
        owner: { connectionId: 'voice-host', profile: 'voice-label' },
        runtimeSessionId: RUNTIME_SESSION_ID,
        storedSessionId: 'voice-stored'
      })
    } else {
      expect(onAssigned).not.toHaveBeenCalled()
      expect(routed).toHaveBeenCalledWith('voice-host', 'voice-label', 'session.close', { session_id: RUNTIME_SESSION_ID })
    }
  })
})
