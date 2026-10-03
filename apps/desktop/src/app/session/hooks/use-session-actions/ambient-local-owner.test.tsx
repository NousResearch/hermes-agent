import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { getLatestSessionMessages, type SessionInfo } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import * as gateways from '@/store/gateway'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import { _resetSessionOwnerHintsForTests, getSessionOwnerHint, setConnection, setSessions } from '@/store/session'
import { clearAllSessionStates, publishSessionState, requestForOwnedSession } from '@/store/session-states'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/app/contrib/hooks/use-background-sync', () => ({ resetLiveRuntimeTracking: vi.fn() }))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'stored' })),
  getProfiles: vi.fn(async () => ({ profiles: [{ name: 'default' }] })),
  HermesGateway: class {
    connectionState = 'closed'
    connect = async () => {
      this.connectionState = 'open'
    }
    close = () => {
      this.connectionState = 'closed'
    }
    onEvent = () => () => undefined
    onState = () => () => undefined
    request = vi.fn(async (method: string) => rpcResult(method))
  }
}))

function rpcResult(method: string) {
  return method === 'session.resume' ? { info: {}, messages: [], resumed: 'stored', session_id: 'runtime' } : {}
}

function descriptor(connectionId: string, registryScoped = true): HermesConnection {
  return {
    connectionId,
    profile: 'default',
    registryScoped,
    mode: connectionId === 'local' ? 'local' : 'remote',
    authMode: 'token',
    token: 'test',
    wsUrl: `ws://${connectionId}.invalid/ws`,
    baseUrl: `http://${connectionId}.invalid`
  } as HermesConnection
}

type Resume = ReturnType<typeof useSessionActions>['resumeSession']
const ambientRequest = vi.fn(async () => ({}) as never)

function Harness({ onReady }: { onReady: (resume: Resume) => void }) {
  const states = useRef(new Map<string, ClientSessionState>())
  const activeSessionIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const selectedStoredSessionIdRef = useRef<string | null>(null)

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: () => createClientSessionState(null),
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway: ambientRequest,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: states,
    syncSessionStateToView: vi.fn(),
    updateSessionState: (id, updater, storedId) => {
      const next = updater(states.current.get(id) ?? createClientSessionState(storedId ?? null))
      states.current.set(id, next)
      publishSessionState(id, next)

      return next
    }
  })

  useEffect(() => onReady(actions.resumeSession), [actions.resumeSession, onReady])

  return null
}

describe('untagged resume ambient owner', () => {
  beforeEach(() => {
    clearAllSessionStates()
    _resetSessionOwnerHintsForTests()
    $showAllProfiles.set(false)
    $activeGatewayProfile.set('default')
    vi.clearAllMocks()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getConnectionFor: vi.fn(async ({ connectionId }: { connectionId: string }) => descriptor(connectionId)),
        getGatewayWsUrlFor: vi.fn(async ({ connectionId }: { connectionId: string }) => ({
          ok: true,
          wsUrl: `ws://${connectionId}.invalid/ws`
        }))
      }
    })
    gateways.configureGatewayRegistry({ onEvent: vi.fn(), onActiveRouteChanged: p => $activeGatewayProfile.set(p) })
  })

  afterEach(() => {
    cleanup()
    gateways.closeSecondaryGateways()
    gateways.setPrimaryGateway(null)
    setConnection(null)
    setSessions([])
    clearAllSessionStates()
    _resetSessionOwnerHintsForTests()
    vi.restoreAllMocks()
  })

  it.each(['local', 'remote-secondary'])(
    'keeps resume, transcript and subsequent prompt on registry secondary %s',
    async connectionId => {
      const primary = { connectionState: 'open', request: vi.fn(async (method: string) => rpcResult(method)) }
      gateways.setPrimaryGateway(primary as never, 'default')
      gateways.setPrimaryGatewayConnection(descriptor('home'))
      setConnection(descriptor('home'))
      await gateways.ensureGatewayForAgent(connectionId, 'default')
      const secondary = gateways.activeGateway()!
      expect(gateways.isActivePrimary()).toBe(false)
      expect(gateways.activeGatewayConnectionId()).toBe(connectionId)
      // Deliberately no row tag or remembered owner: this is the ambient fallback.
      setSessions([{ id: 'stored', profile: 'default', message_count: 0 } as SessionInfo])
      expect(getSessionOwnerHint('stored')).toBeUndefined()
      let resume!: Resume
      render(
        <Harness
          onReady={ready => {
            resume = ready
          }}
        />
      )
      await act(() => resume('stored', true))

      expect(secondary.request).toHaveBeenCalledWith(
        'session.resume',
        expect.objectContaining({ session_id: 'stored' })
      )
      expect(getLatestSessionMessages).toHaveBeenCalledWith('stored', { connectionId, profile: 'default' })
      await gateways.ensureGatewayForAgent('home', 'default')
      await requestForOwnedSession('runtime', ambientRequest, 'prompt.submit', {
        session_id: 'runtime',
        text: 'continue'
      })
      expect(secondary.request).toHaveBeenCalledWith('prompt.submit', { session_id: 'runtime', text: 'continue' })
      expect(primary.request).not.toHaveBeenCalled()
      expect(ambientRequest).not.toHaveBeenCalled()
    }
  )

  it.each([false, true])(
    'preserves the profile-only door on a local primary (registryScoped=%s)',
    async registryScoped => {
      const primary = { connectionState: 'open', request: vi.fn(async (method: string) => rpcResult(method)) }
      gateways.setPrimaryGateway(primary as never, 'default')
      gateways.setPrimaryGatewayConnection(descriptor('local', registryScoped))
      setConnection(descriptor('local', registryScoped))
      const profileRequest = vi.spyOn(gateways, 'requestGatewayForProfile')
      const agentRequest = vi.spyOn(gateways, 'requestGatewayForAgent')
      setSessions([{ id: 'stored', profile: 'default', message_count: 0 } as SessionInfo])
      let resume!: Resume
      render(
        <Harness
          onReady={ready => {
            resume = ready
          }}
        />
      )
      await act(() => resume('stored', true))

      expect(profileRequest).toHaveBeenCalledWith(
        'default',
        'session.resume',
        expect.objectContaining({ session_id: 'stored' }),
        undefined,
        undefined
      )
      expect(agentRequest).not.toHaveBeenCalled()
      expect(getLatestSessionMessages).toHaveBeenCalledWith('stored', 'default')
      expect(primary.request).toHaveBeenCalledWith('session.resume', expect.objectContaining({ session_id: 'stored' }))
    }
  )
})
