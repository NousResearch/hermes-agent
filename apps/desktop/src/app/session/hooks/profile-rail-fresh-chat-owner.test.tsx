import { registryBackendScopeKey } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useMemo, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, type Mock, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { getSession } from '@/hermes'
import {
  activeGateway,
  activeGatewayConnectionId,
  activeGatewayProfileKey,
  closeSecondaryGateways,
  configureGatewayRegistry,
  setPrimaryGateway
} from '@/store/gateway'
import {
  $activeGatewayProfile,
  $newChatConnectionId,
  $newChatProfile,
  $newChatRoute,
  ensureGatewayAgent,
  selectProfile
} from '@/store/profile'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  $sessions,
  getSessionOwnerHint,
  sessionMatchesStoredId,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'

import type { ClientSessionState } from '../../types'

import { usePromptActions } from './use-prompt-actions'
import { clearSingleFlightSessionResumeState } from './use-prompt-actions/single-flight-resume'
import type { SubmitTextOptions } from './use-prompt-actions/utils'
import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

// ── The real profile-rail reproduction (#94071, Sessions mode) ───────────────
//
//   primary / ambient source  = a remote gateway on `default`
//   active registry source    = `homelab` (a remote registry source)
//   user action               = selectProfile("omar") in the profile rail
//
// selectProfile sets $newChatProfile = "omar" and deliberately CLEARS
// $newChatRoute, so nothing explicit names the source. The draft's real owner
// is the registry entry homelab::omar (scope `conn:homelab::omar`) — the
// socket whose WebSocket mints the runtime. Before the fix the create rode
// that socket ambiently, but the durable owner degraded to the bare string
// "omar": the optimistic row was stamped from the ambient profile with no
// connection, no owner hint was recorded, and every follow-up RPC dialed
// requestGatewayForProfile("omar") — a DIFFERENT v1 socket/backend that never
// held the runtime — and 4001'd "session not found" while the orphaned omar
// runtime was left to be ws-orphan-reaped.
//
// The explicit `local` source (This device) is different by design: a profile
// pick made there takes the legacy profile-only door (ensureGatewayProfile,
// so a per-profile remote override still resolves), and the draft's owner is
// that v1 profile socket — the second case pins that the same one-socket
// continuity holds there too.
//
// This suite drives the ACTUAL code path: the real registry store with mocked
// sockets, the real store/profile switch, the real useSessionStateCache /
// useSessionActions / usePromptActions hooks, and the production session-RPC
// dispatcher. It never supplies an owner by hand.

const SOURCE_ID = 'homelab'
const OMAR_PORT = 7171
const SOURCE_DEFAULT_PORT = 7070
const V1_PORT = 5151
const RUNTIME_ID = 'rt-omar-fresh-1'
const STORED_ID = 'stored-omar-fresh-1'

type GatewayRequestMock = Mock<(method: string, params?: Record<string, unknown>) => Promise<unknown>>

interface MockGateway {
  connectUrl: null | string
  connectionState: string
  connect: Mock<(url: string) => Promise<void>>
  close: Mock<() => void>
  onEvent: Mock<() => () => void>
  onState: Mock<() => () => void>
  request: GatewayRequestMock
}

const sockets: MockGateway[] = []
/** The port of the ONE socket allowed to mint (and then own) the runtime. */
let ownerPort = OMAR_PORT
/** The ids the owner socket mints — per case, so one case's owner records
 *  (the hint map is module state) can never satisfy another's assertions. */
let mintedRuntimeId = RUNTIME_ID
let mintedStoredId = STORED_ID

const sessionScoped = (params: unknown) =>
  typeof (params as { session_id?: unknown } | undefined)?.session_id === 'string'

/** The owner socket (the registry entry homelab::omar, or the v1 omar socket
 *  for a legacy pick) answers; every other socket is a backend that never
 *  held the runtime, exactly as in the field. */
function answer(socket: MockGateway, method: string, params: Record<string, unknown>) {
  const isOmar = socket.connectUrl?.includes(`:${ownerPort}`) ?? false

  if (method === 'session.create') {
    if (!isOmar) {
      throw new Error(`session.create landed on the wrong socket: ${socket.connectUrl}`)
    }

    return { info: {}, session_id: mintedRuntimeId, stored_session_id: mintedStoredId }
  }

  if (sessionScoped(params) && !isOmar) {
    throw new Error(`Session not found: ${String(params.session_id)} (socket ${socket.connectUrl}, ${method})`)
  }

  if (method === 'prompt.submit') {
    return { ok: true }
  }

  if (method === 'session.resume' || method === 'session.activate') {
    // The runtime is alive on this socket: a resume re-binds the SAME id.
    return {
      info: {},
      message_count: 1,
      messages: [],
      resumed: mintedStoredId,
      running: false,
      session_id: mintedRuntimeId,
      session_key: mintedStoredId
    }
  }

  return {}
}

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  HermesGateway: class {
    connectUrl: null | string = null
    connectionState = 'closed'
    connect = vi.fn(async (url: string) => {
      this.connectUrl = url
      this.connectionState = 'open'
    })
    request = vi.fn(async (method: string, params: Record<string, unknown> = {}) => {
      if (this.connectionState !== 'open') {
        throw new Error('gateway is not connected')
      }

      return answer(this as unknown as MockGateway, method, params)
    })
    close = vi.fn(() => {
      this.connectionState = 'closed'
    })
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})

    constructor() {
      sockets.push(this as unknown as MockGateway)
    }
  },
  getSession: vi.fn(async () => {
    throw new Error('REST cross-profile probe must not be needed: the owner is known')
  }),
  setApiRequestConnection: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    // v1 profile path (requestGatewayForProfile / ensureGatewayProfile): a
    // per-profile local backend that is NOT the registry entry.
    getConnection: vi.fn(async (profile: null | string) => {
      const port = profile ? V1_PORT : 4242

      return { port, profile, token: profile ? 'v1-token' : 'primary-token', wsUrl: `ws://127.0.0.1:${port}/ws` }
    }),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => {
      const port =
        connectionId === SOURCE_ID || connectionId === 'local'
          ? profile === 'omar'
            ? OMAR_PORT
            : SOURCE_DEFAULT_PORT
          : 9999

      return { port, profile, token: `${connectionId}-${profile}-token`, wsUrl: `ws://127.0.0.1:${port}/ws` }
    }),
    touchBackend: vi.fn(async () => undefined)
  }
}

/** The remote primary. Session-scoped traffic here is the bug. */
function makePrimary(): MockGateway {
  const primary: MockGateway = {
    connectUrl: 'ws://remote-primary:4242',
    connectionState: 'open',
    connect: vi.fn(),
    close: vi.fn(),
    onEvent: vi.fn(() => () => {}),
    onState: vi.fn(() => () => {}),
    request: vi.fn(async (method: string, params: Record<string, unknown> = {}) => answer(primary, method, params))
  }

  return primary
}

interface HarnessHandle {
  busyRef: { current: boolean }
  bindings: () => { runtimeForStored: null | string; storedForRuntime: null | string }
  submitText: (text: string, options?: SubmitTextOptions) => Promise<boolean>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => ClientSessionState
}

/** The window's real hook stack, wired the way contrib/wiring wires it. */
function Harness({
  ambientRequest,
  onReady
}: {
  ambientRequest: MockGateway['request']
  onReady: (h: HarnessHandle) => void
}) {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const requestGateway = useMemo(
    () =>
      createSessionRpcDispatcher({
        ambientRequest: ambientRequest as never,
        runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
        selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
        sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef
      }),
    [
      ambientRequest,
      cache.runtimeIdByStoredSessionIdRef,
      cache.selectedStoredSessionIdRef,
      cache.sessionStateByRuntimeIdRef
    ]
  )

  const sessionActions = useSessionActions({
    activeSessionId,
    activeSessionIdRef: cache.activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: cache.ensureSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: cache.resetViewSync,
    runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef,
    syncSessionStateToView: cache.syncSessionStateToView,
    updateSessionState: cache.updateSessionState
  })

  const promptActions = usePromptActions({
    activeSessionId,
    activeSessionIdRef: cache.activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef,
    createBackendSessionForSend: sessionActions.createBackendSessionForSend,
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: cache.getRuntimeIdForStoredSession,
    getRouteToken: () => 'token',
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: sessionActions.resumeSession,
    runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
    startFreshSessionDraft: sessionActions.startFreshSessionDraft,
    sttEnabled: false,
    updateSessionState: cache.updateSessionState
  })

  const { submitText } = promptActions

  useEffect(() => {
    onReady({
      busyRef,
      bindings: () => ({
        runtimeForStored: cache.runtimeIdByStoredSessionIdRef.current.get(mintedStoredId) ?? null,
        storedForRuntime: cache.sessionStateByRuntimeIdRef.current.get(mintedRuntimeId)?.storedSessionId ?? null
      }),
      submitText: (...args) => act(async () => submitText(...args)) as Promise<boolean>,
      updateSessionState: cache.updateSessionState as HarnessHandle['updateSessionState']
    })
  }, [
    cache.runtimeIdByStoredSessionIdRef,
    cache.sessionStateByRuntimeIdRef,
    cache.updateSessionState,
    onReady,
    submitText
  ])

  return null
}

const omarScope = registryBackendScopeKey(SOURCE_ID, 'omar')

describe('profile rail: a fresh Omar chat keeps its exact registry owner across turns (#94071)', () => {
  beforeEach(() => {
    sockets.length = 0
    ownerPort = OMAR_PORT
    mintedRuntimeId = RUNTIME_ID
    mintedStoredId = STORED_ID
    clearSingleFlightSessionResumeState()
    configureGatewayRegistry({ onEvent: vi.fn() })
    closeSecondaryGateways()
    installDesktop()
    setSessions([])
    setMessages([])
    setActiveSessionId(null)
    setSelectedStoredSessionId(null)
    setBusy(false)
    setAwaitingResponse(false)
    $newChatProfile.set(null)
    $newChatRoute.set(null)
    $newChatConnectionId.set(null)
  })

  afterEach(() => {
    cleanup()
    closeSecondaryGateways()
    setSessions([])
    setActiveSessionId(null)
    setSelectedStoredSessionId(null)
    $newChatProfile.set(null)
    $newChatRoute.set(null)
    $newChatConnectionId.set(null)
    $activeGatewayProfile.set('default')
    vi.clearAllMocks()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('session.create and both prompt.submit calls ride the SAME conn:homelab::omar socket', async () => {
    // Primary / ambient source: a remote gateway on `default`.
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')

    // Active registry source: `homelab` (a remote source), on its default
    // profile — the state a connection-rail click leaves the window in.
    await ensureGatewayAgent(SOURCE_ID, 'default')
    expect(activeGatewayConnectionId()).toBe(SOURCE_ID)

    // The profile rail: selectProfile("omar").
    selectProfile('omar')
    expect($newChatProfile.get()).toBe('omar')
    expect($newChatRoute.get()).toBeNull()
    await waitFor(() => expect(activeGatewayProfileKey()).toBe('omar'))
    expect(activeGatewayConnectionId()).toBe(SOURCE_ID)

    // The socket the registry dialed for homelab::omar (mocked HermesGateway
    // instances register themselves on construction).
    expect(sockets.length).toBeGreaterThan(0)
    const omarSocket = sockets.find(socket => socket.connectUrl?.includes(`:${OMAR_PORT}`))
    expect(
      omarSocket,
      `no socket dialed port ${OMAR_PORT}; dialed: ${sockets.map(s => s.connectUrl).join(', ')}`
    ).toBeDefined()
    expect(activeGateway()).toBe(omarSocket as never)

    // Ambient dispatcher = whatever socket is active, as useGatewayRequest does.
    const ambientRequest = vi.fn(async (method: string, params?: Record<string, unknown>) =>
      (activeGateway() as unknown as MockGateway).request(method, params)
    )

    let handle: HarnessHandle | null = null
    render(<Harness ambientRequest={ambientRequest as never} onReady={h => (handle = h)} />)
    await waitFor(() => expect(handle).not.toBeNull())

    // Turn one: no session yet → createBackendSessionForSend → prompt.submit.
    await expect(handle!.submitText('first prompt')).resolves.toBe(true)
    await waitFor(() => expect($activeSessionId.get()).toBe(RUNTIME_ID))

    // The stored↔runtime binding minted by the create must survive the first
    // turn: submit used to seed its optimistic bubble with the PRE-create
    // (null) stored id, which the state cache read as a detach — after which
    // no session-scoped RPC could translate the runtime id back to the stored
    // id, so tile route / owner hint / row were all bypassed.
    expect(handle!.bindings()).toEqual({ runtimeForStored: RUNTIME_ID, storedForRuntime: STORED_ID })

    // Answer one arrives: the turn settles (what the gateway's stream end does).
    await act(async () => {
      handle!.updateSessionState(RUNTIME_ID, state => ({
        ...state,
        awaitingResponse: false,
        busy: false,
        streamId: null,
        turnStartedAt: null
      }))
      handle!.busyRef.current = false
      setBusy(false)
      setAwaitingResponse(false)
    })

    // Turn two on the now-existing session.
    await expect(handle!.submitText('second prompt')).resolves.toBe(true)
    expect(handle!.bindings()).toEqual({ runtimeForStored: RUNTIME_ID, storedForRuntime: STORED_ID })

    // Every session-scoped RPC (create + both submits) hit ONE socket: the
    // registry entry conn:homelab::omar that minted the runtime.
    const calls = (socket: MockGateway) => socket.request.mock.calls.map(call => call[0] as string)
    const omarCalls = calls(omarSocket!)

    expect(omarCalls).toContain('session.create')
    expect(omarCalls.filter(method => method === 'prompt.submit')).toHaveLength(2)
    expect(
      omarSocket!.request.mock.calls
        .filter(call => call[0] === 'prompt.submit')
        .map(call => [(call[1] as { session_id: string }).session_id, (call[1] as { text: string }).text])
    ).toEqual([
      [RUNTIME_ID, 'first prompt'],
      [RUNTIME_ID, 'second prompt']
    ])
    expect(activeGateway()).toBe(omarSocket as never)
    expect(registryBackendScopeKey(activeGatewayConnectionId(), activeGatewayProfileKey())).toBe(omarScope)

    // Nothing session-scoped reached the remote primary, the source's default
    // socket, or a v1 requestGatewayForProfile("omar") socket.
    expect(calls(primary).filter(method => method === 'session.create' || method === 'prompt.submit')).toEqual([])

    for (const socket of sockets) {
      if (socket !== omarSocket) {
        expect(
          socket.request.mock.calls.filter(call => sessionScoped(call[1]) || call[0] === 'session.create')
        ).toEqual([])
      }
    }

    expect(sockets.some(socket => socket.connectUrl?.includes(`:${V1_PORT}`))).toBe(false)

    // No session-not-found: any misrouted RPC would have thrown out of
    // submitText (asserted true above) — and no REST probe was needed.
    expect(vi.mocked(getSession)).not.toHaveBeenCalled()

    // No ws_orphan_reap precondition: the client never closed, re-created or
    // abandoned the runtime it minted; the durable owner is the exact entry.
    for (const socket of [primary, ...sockets]) {
      expect(calls(socket).filter(method => method === 'session.close')).toEqual([])
    }

    expect(omarCalls.filter(method => method === 'session.create')).toHaveLength(1)
    expect(getSessionOwnerHint(STORED_ID)).toEqual({ connectionId: SOURCE_ID, profile: 'omar' })
    expect($sessions.get().find(session => sessionMatchesStoredId(session, STORED_ID))).toMatchObject({
      connection_id: SOURCE_ID,
      profile: 'omar'
    })
    expect($newChatConnectionId.get()).toBe(SOURCE_ID)
  })

  it('a pick on the explicit `local` source is a legacy profile pick: create and both turns ride the ONE v1 omar socket', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')

    // Active registry source: `local` (This device), on its default profile.
    await ensureGatewayAgent('local', 'default')
    expect(activeGatewayConnectionId()).toBe('local')

    // A profile pick on the explicit local source takes the profile-only door
    // so a per-profile remote override resolves (the main process answers
    // getConnection("omar")), never the registry entry local::omar. The draft's
    // owner must be the socket that door opens: the v1 omar socket.
    const LEGACY_RUNTIME_ID = 'rt-omar-legacy-1'
    const LEGACY_STORED_ID = 'stored-omar-legacy-1'

    ownerPort = V1_PORT
    mintedRuntimeId = LEGACY_RUNTIME_ID
    mintedStoredId = LEGACY_STORED_ID
    selectProfile('omar')
    expect($newChatProfile.get()).toBe('omar')
    expect($newChatRoute.get()).toBeNull()
    expect($newChatConnectionId.get()).toBeNull()
    await waitFor(() => expect(activeGatewayProfileKey()).toBe('omar'))
    expect(activeGatewayConnectionId()).toBeNull()

    const desktop = window.hermesDesktop!

    expect(desktop.getConnection).toHaveBeenCalledWith('omar')
    expect(desktop.getConnectionFor).not.toHaveBeenCalledWith({ connectionId: 'local', profile: 'omar' })

    const v1Socket = sockets.find(socket => socket.connectUrl?.includes(`:${V1_PORT}`))
    expect(v1Socket, `no v1 socket dialed; dialed: ${sockets.map(s => s.connectUrl).join(', ')}`).toBeDefined()
    expect(activeGateway()).toBe(v1Socket as never)
    expect(sockets.some(socket => socket.connectUrl?.includes(`:${OMAR_PORT}`))).toBe(false)

    const ambientRequest = vi.fn(async (method: string, params?: Record<string, unknown>) =>
      (activeGateway() as unknown as MockGateway).request(method, params)
    )

    let handle: HarnessHandle | null = null
    render(<Harness ambientRequest={ambientRequest as never} onReady={h => (handle = h)} />)
    await waitFor(() => expect(handle).not.toBeNull())

    await expect(handle!.submitText('first prompt')).resolves.toBe(true)
    await waitFor(() => expect($activeSessionId.get()).toBe(LEGACY_RUNTIME_ID))
    expect(handle!.bindings()).toEqual({ runtimeForStored: LEGACY_RUNTIME_ID, storedForRuntime: LEGACY_STORED_ID })

    await act(async () => {
      handle!.updateSessionState(LEGACY_RUNTIME_ID, state => ({
        ...state,
        awaitingResponse: false,
        busy: false,
        streamId: null,
        turnStartedAt: null
      }))
      handle!.busyRef.current = false
      setBusy(false)
      setAwaitingResponse(false)
    })

    await expect(handle!.submitText('second prompt')).resolves.toBe(true)

    // The legacy owner is the bare profile: no registry route, no hint — the
    // row's profile names the same v1 pool entry that minted the runtime.
    const calls = (socket: MockGateway) => socket.request.mock.calls.map(call => call[0] as string)

    expect(calls(v1Socket!).filter(method => method === 'session.create')).toHaveLength(1)
    expect(
      v1Socket!.request.mock.calls
        .filter(call => call[0] === 'prompt.submit')
        .map(call => (call[1] as { text: string }).text)
    ).toEqual(['first prompt', 'second prompt'])
    expect(calls(primary).filter(method => method === 'session.create' || method === 'prompt.submit')).toEqual([])

    for (const socket of sockets) {
      if (socket !== v1Socket) {
        expect(
          socket.request.mock.calls.filter(call => sessionScoped(call[1]) || call[0] === 'session.create')
        ).toEqual([])
      }
    }

    expect(sockets.some(socket => socket.connectUrl?.includes(`:${OMAR_PORT}`))).toBe(false)
    expect(vi.mocked(getSession)).not.toHaveBeenCalled()

    for (const socket of [primary, ...sockets]) {
      expect(calls(socket).filter(method => method === 'session.close')).toEqual([])
    }

    expect(getSessionOwnerHint(LEGACY_STORED_ID)).toBeUndefined()
    expect($sessions.get().find(session => sessionMatchesStoredId(session, LEGACY_STORED_ID))).toMatchObject({
      profile: 'omar'
    })
  })
})
