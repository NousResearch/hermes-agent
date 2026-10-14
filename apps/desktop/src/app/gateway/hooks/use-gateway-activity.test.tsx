import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { renderMessageStream } from '@/app/session/hooks/use-message-stream/test-harness'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $desktopBoot } from '@/store/boot'
import {
  closeSecondaryGateways,
  ensureGatewayForAgent,
  openGatewayForAgent,
  requestGatewayForAgent,
  retainGatewayForRelay,
  retainGatewayForSessionTurn,
  type ScopedServerRequest,
  setPrimaryGateway
} from '@/store/gateway'
import { $goalsBySession, setSessionGoal } from '@/store/goals'
import { $activeGatewayProfile, ensureGatewayAgent, ensureGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { clearAllSessionStates, publishSessionState, runtimeSessionOwner } from '@/store/session-states'
import { $subagentsBySession } from '@/store/subagents'
import { FakeWebSocket } from '@/test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

// Real boot, message stream, registry and JSON-RPC client. Only Electron IPC
// and the network boundary are faked; no backend or model is started.
class ActivitySocket extends FakeWebSocket {
  deferApprovals = false
  approvals: Array<{ id?: unknown; params?: { session_id?: string } }> = []
  private messages = new Set<(event: unknown) => void>()
  override addEventListener(type: string, listener: (event: unknown) => void) {
    super.addEventListener(type, listener)

    if (type === 'message') {
      this.messages.add(listener)
    }
  }
  override removeEventListener(type: string, listener: (event: unknown) => void) {
    super.removeEventListener(type, listener)

    if (type === 'message') {
      this.messages.delete(listener)
    }
  }
  override send(data: string) {
    const frame = JSON.parse(data) as { id?: unknown; method?: string; params?: { session_id?: string } }

    // Idle session.info also polls approvals. Answer the fake network request
    // so an unrelated in-flight RPC cannot masquerade as settlement retention.
    if (frame.method === 'approval.pending') {
      this.approvals.push(frame)

      if (!this.deferApprovals) {
        this.replyApproval(frame.id)
      }

      return
    }

    super.send(data)
  }
  replyApproval(id: unknown) {
    for (const listener of this.messages) {
      listener({ data: JSON.stringify({ jsonrpc: '2.0', id, result: { approvals: [] } }) })
    }
  }
  request() {
    for (const listener of this.messages) {
      listener({ data: JSON.stringify({ jsonrpc: '2.0', id: 'approval', method: 'clarify', params: {} }) })
    }
  }
  receive(event: GatewayEvent) {
    for (const listener of this.messages) {
      listener({ data: JSON.stringify({ jsonrpc: '2.0', method: 'event', params: event }) })
    }
  }
}

const descriptor = (connectionId: string, profile = 'default') => ({
  authMode: 'token' as const,
  baseUrl: `https://${connectionId}.invalid`,
  isFullscreen: false,
  nativeOverlayWidth: 0,
  windowButtonPosition: null,
  logs: [],
  connectionId,
  mode: 'remote' as const,
  profile,
  token: 'test-only',
  wsUrl: `wss://${connectionId}.invalid/api/ws?profile=${profile}`
})

const handleServerRequest = vi.fn((_request: ScopedServerRequest) => false)
const primary = descriptor('primary')
const touchBackend = vi.fn(async (_scope: string, _options?: { activeTurn?: boolean }) => ({ ok: true }))

function Harness({ onEvent }: { onEvent: (event: GatewayEvent) => void }) {
  useGatewayBoot({
    beforeConnectionSwitch: () => undefined,
    handleGatewayEvent: onEvent,
    handleServerRequest,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined
  })

  return null
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2026-01-01T00:00:00Z'))
  vi.stubGlobal('WebSocket', ActivitySocket)
  FakeWebSocket.instances = []
  FakeWebSocket.mode = 'open'
  FakeWebSocket.pingMode = 'pong'
  touchBackend.mockClear()
  handleServerRequest.mockClear()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $gatewayState.set('idle')
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: '',
    phase: 'init',
    progress: 0,
    running: true,
    timestamp: Date.now(),
    visible: true
  })

  const unsubscribe = () => () => undefined

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
    getConnection: vi.fn(async () => primary),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
      descriptor(connectionId, profile)
    ),
    getGatewayWsUrl: vi.fn(async (conn: { wsUrl: string }) => conn.wsUrl),
    getGatewayWsUrlFor: vi.fn(async (conn: { wsUrl: string }) => conn.wsUrl),
    getBootProgress: vi.fn(async () => ({ ...$desktopBoot.get(), retryable: false })),
    onBootProgress: unsubscribe,
    onBackendExit: unsubscribe,
    onConnectionApplied: unsubscribe,
    onPowerResume: unsubscribe,
    onWindowStateChanged: unsubscribe,
    touchBackend,
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
})

afterEach(() => {
  cleanup()
  takeGatewaySurvivor()?.gateway.close()
  closeSecondaryGateways()
  setPrimaryGateway(null)
  clearAllSessionStates()
  $subagentsBySession.set({})
  $goalsBySession.set({})
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

async function boot() {
  const states = new Map<string, ClientSessionState>()

  const stream = renderMessageStream(null, {
    states,
    updateSessionState: (id, update) => {
      const next = update(states.get(id) ?? createClientSessionState())
      states.set(id, next)
      publishSessionState(id, next)

      return next
    }
  })

  render(<Harness onEvent={stream.handleEvent} />)
  await advance(1)
  expect($gatewayState.get()).toBe('open')
}

function socketFor(connectionId: string): ActivitySocket {
  return FakeWebSocket.instances.findLast(socket =>
    socket.url.startsWith(`wss://${connectionId}.invalid/`)
  ) as ActivitySocket
}

function emit(
  socket: ActivitySocket,
  type: GatewayEvent['type'],
  payload: Record<string, unknown> = {},
  sessionId = 'automatic-session'
) {
  act(() => socket.receive({ type, session_id: sessionId, payload }))
}

it.each([
  { connectionId: null, work: 'child', reply: 'before' },
  { connectionId: null, work: 'child', reply: 'after' },
  { connectionId: null, work: 'session', reply: 'before' },
  { connectionId: null, work: 'session', reply: 'after' },
  { connectionId: 'remote', work: 'child', reply: 'before' },
  { connectionId: 'remote', work: 'child', reply: 'after' },
  { connectionId: 'remote', work: 'session', reply: 'before' },
  { connectionId: 'remote', work: 'session', reply: 'after' }
])(
  'direct retain keeps $connectionId/$work with approval reply $reply parent release',
  async ({ connectionId, work, reply }) => {
    vi.mocked(window.hermesDesktop!.getConnection).mockImplementation(async profile =>
      profile === 'writer' ? { ...descriptor('profile', profile), connectionId: undefined, mode: 'local' } : primary
    )
    await boot()
    // No hover/open/activation: this lease alone creates the secondary.
    const retained = retainGatewayForSessionTurn(connectionId, 'writer', 'automatic-session')
    await advance(1)
    const release = await retained
    const socket = socketFor(connectionId ?? 'profile')
    const scope = connectionId ? 'conn:remote::writer' : 'writer'
    const owner = connectionId ? { connectionId, profile: 'writer' } : 'writer'
    socket.deferApprovals = reply === 'after'
    emit(socket, 'message.start')

    if (work === 'child') {
      emit(socket, 'subagent.start', { subagent_id: 'child', status: 'running', goal: 'work' })
    } else {
      emit(socket, 'message.start', {}, 'another-session')
    }

    emit(socket, 'message.complete', { text: 'waiting' })
    emit(socket, 'session.info', { running: false })
    await advance(1)
    expect(socket.approvals).toHaveLength(1)
    expect(socket.approvals[0].id).toBeDefined()
    expect(socket.approvals[0].params?.session_id).toBe('automatic-session')
    expect(socketFor('primary').approvals).toEqual([])
    touchBackend.mockClear()
    await advance(500)
    expect(socket.readyState).toBe(FakeWebSocket.OPEN)
    expect(touchBackend).toHaveBeenCalledWith(scope, { activeTurn: true })
    expect(touchBackend).not.toHaveBeenCalledWith(scope, { activeTurn: false })

    if (reply === 'after') {
      act(() => socket.replyApproval(socket.approvals[0].id))
      await advance(1)
    }

    // Request-finally must not undo retention after the parent hold is gone.
    expect(socket.readyState).toBe(FakeWebSocket.OPEN)
    expect(touchBackend).not.toHaveBeenCalledWith(scope, { activeTurn: false })
    expect(runtimeSessionOwner('automatic-session')).toEqual(owner)
    expect($activeGatewayProfile.get()).toBe('default')
    expect($connection.get()?.connectionId).toBe('primary')
    touchBackend.mockClear()
    release()
    expect(touchBackend).not.toHaveBeenCalled()
    await advance(61 * 60_000)
    expect(socket.readyState).toBe(FakeWebSocket.OPEN)
    expect(touchBackend).toHaveBeenCalledWith(scope, { activeTurn: true })
    expect(touchBackend).not.toHaveBeenCalledWith(scope, { activeTurn: false })
    socket.deferApprovals = false

    if (work === 'child') {
      emit(socket, 'subagent.complete', { subagent_id: 'child', status: 'completed' })
    } else {
      // Settle directly from busy: message.complete would already clear it,
      // leaving no live-to-idle edge for session.info's handoff window.
      emit(socket, 'session.info', { running: false }, 'another-session')
      expect(runtimeSessionOwner('another-session')).toEqual(owner)
    }

    expect(touchBackend).toHaveBeenLastCalledWith(scope, { activeTurn: false })
    await advance(499)
    expect(socket.readyState).toBe(FakeWebSocket.OPEN)
    await advance(60_000)
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
    expect(touchBackend.mock.calls.filter(([key]) => key === scope).at(-1)?.[1]).toEqual({ activeTurn: false })
    expect(FakeWebSocket.instances).toHaveLength(2)
  }
)

it('relay release keeps authoritative work without a turn lease, then reclaims idle', async () => {
  await boot()
  const release = retainGatewayForRelay('remote', 'writer')
  const request = requestGatewayForAgent('remote', 'writer', 'ping')
  await advance(1)
  await expect(request).resolves.toEqual({ pong: true })
  const socket = socketFor('remote')
  emit(socket, 'message.start')
  touchBackend.mockClear()
  release()
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)
  expect(touchBackend).not.toHaveBeenCalledWith('conn:remote::writer', { activeTurn: false })
  await advance(61 * 60_000)
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)
  expect(runtimeSessionOwner('automatic-session')).toEqual({ connectionId: 'remote', profile: 'writer' })
  emit(socket, 'session.info', { running: false })
  await advance(499)
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)
  await advance(60_000)
  expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
  expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: false })
  expect(FakeWebSocket.instances).toHaveLength(2)
})

it.each(['message.start', 'session.info'] as const)(
  'accounts for a delayed %s after the submitted turn lease settled',
  async type => {
    await boot()
    const opened = openGatewayForAgent('remote', 'writer')
    await advance(1)
    await opened
    await retainGatewayForSessionTurn('remote', 'writer', 'automatic-session')
    const socket = socketFor('remote')
    emit(socket, 'message.start')
    emit(socket, 'message.complete', { text: 'waiting' })
    emit(socket, 'session.info', { running: false })
    await advance(501)
    expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: false })

    // A server-originated continuation has no renderer prompt.submit lease.
    emit(socket, type, type === 'session.info' ? { running: true } : {})
    expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: true })
    touchBackend.mockClear()
    await advance(61 * 60_000)
    const touches = touchBackend.mock.calls.filter(([scope]) => scope === 'conn:remote::writer')
    expect(touches.length).toBeGreaterThan(60)
    expect(touches.every(([, options]) => options?.activeTurn === true)).toBe(true)
    expect(socket.readyState).toBe(FakeWebSocket.OPEN)

    emit(socket, 'message.complete', { text: 'done' })
    emit(socket, 'session.info', { running: false })
    await advance(60_000)
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
    expect(touchBackend.mock.calls.filter(([scope]) => scope === 'conn:remote::writer').at(-1)?.[1]).toEqual({
      activeTurn: false
    })
  }
)

it.each(['default', 'reader'])(
  'keeps a pooled primary attributed to its owner while another connection/%s is foregrounded',
  async foregroundProfile => {
    await boot()
    const socket = socketFor('primary')
    emit(socket, 'message.start')
    await advance(60_000)
    expect(touchBackend).toHaveBeenLastCalledWith('conn:primary::default', { activeTurn: true })

    const switched = ensureGatewayForAgent('other', foregroundProfile)
    await advance(1)
    await switched
    act(() => socket.request())
    expect(handleServerRequest).toHaveBeenLastCalledWith(
      expect.objectContaining({ connectionId: 'primary', profile: 'default' })
    )
    // The primary is now in the background. New authoritative events must not
    // be attributed to the foreground connection, even with identical profiles.
    emit(socket, 'session.info', { running: true })
    expect(runtimeSessionOwner('automatic-session')).toEqual({ connectionId: 'primary', profile: 'default' })
    touchBackend.mockClear()
    await advance(61 * 60_000)
    expect(
      touchBackend.mock.calls
        .filter(([scope]) => scope === 'conn:primary::default')
        .every(([, options]) => options?.activeTurn === true)
    ).toBe(true)
    expect(touchBackend).toHaveBeenCalledWith('conn:primary::default', { activeTurn: true })
    expect(touchBackend).toHaveBeenCalledWith(`conn:other::${foregroundProfile}`, { activeTurn: false })
    expect(touchBackend).not.toHaveBeenCalledWith('default', expect.anything())
    expect(touchBackend).not.toHaveBeenCalledWith(`conn:other::${foregroundProfile}`, { activeTurn: true })

    // A background primary reconnect must keep the same activity owner.
    act(() => socket.drop())
    await advance(8_000)
    const reconnected = socketFor('primary')
    expect(reconnected).not.toBe(socket)
    expect(reconnected.readyState).toBe(FakeWebSocket.OPEN)
    emit(reconnected, 'session.info', { running: true })
    expect(touchBackend).toHaveBeenLastCalledWith('conn:primary::default', { activeTurn: true })
    expect(runtimeSessionOwner('automatic-session')).toEqual({ connectionId: 'primary', profile: 'default' })

    emit(reconnected, 'message.complete', { text: 'done' })
    emit(reconnected, 'session.info', { running: false })
    touchBackend.mockClear()
    await advance(60_000)
    expect(touchBackend).toHaveBeenCalledWith('conn:primary::default', { activeTurn: false })
  }
)

it('keeps an immediate running-state continuation active when the submit lease releases', async () => {
  await boot()
  const opened = openGatewayForAgent('remote', 'writer')
  await advance(1)
  await opened
  await retainGatewayForSessionTurn('remote', 'writer', 'automatic-session')
  const socket = socketFor('remote')
  emit(socket, 'message.start')
  emit(socket, 'session.info', { running: false })
  emit(socket, 'session.info', { running: true })
  touchBackend.mockClear()
  await advance(501)
  expect(touchBackend).not.toHaveBeenCalledWith('conn:remote::writer', { activeTurn: false })
})

it('does not turn an idle connection or saved active goal into current work', async () => {
  await boot()
  act(() => setSessionGoal('automatic-session', { status: 'active', title: 'Saved goal', updatedAt: Date.now() }))
  emit(socketFor('primary'), 'session.info', { running: false })
  await advance(61 * 60_000)
  expect(touchBackend.mock.calls.length).toBeGreaterThan(60)
  expect(touchBackend.mock.calls.every(([, options]) => options?.activeTurn === false)).toBe(true)
})

it('retains delegated work after the parent settles, then releases the idle route', async () => {
  await boot()
  const opened = openGatewayForAgent('remote', 'writer')
  await advance(1)
  await opened
  const socket = socketFor('remote')
  emit(socket, 'message.start')
  emit(socket, 'subagent.start', { subagent_id: 'child', status: 'running', goal: 'background work' })
  emit(socket, 'message.complete', { text: 'waiting for child' })
  emit(socket, 'session.info', { running: false })
  touchBackend.mockClear()
  await advance(61 * 60_000)
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)
  const touches = touchBackend.mock.calls.filter(([scope]) => scope === 'conn:remote::writer')
  expect(touches.length).toBeGreaterThan(60)
  expect(touches.every(([, options]) => options?.activeTurn === true)).toBe(true)

  emit(socket, 'subagent.complete', { subagent_id: 'child', status: 'completed' })
  await advance(60_000)
  expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
  expect(touchBackend.mock.calls.filter(([scope]) => scope === 'conn:remote::writer').at(-1)?.[1]).toEqual({
    activeTurn: false
  })
})

it.each(['child', 'session'] as const)('releasing a parent lease accounts for another live %s', async work => {
  await boot()
  const opened = openGatewayForAgent('remote', 'writer')
  await advance(1)
  await opened
  const release = await retainGatewayForSessionTurn('remote', 'writer', 'automatic-session')
  const socket = socketFor('remote')
  emit(socket, 'message.start')

  if (work === 'child') {
    emit(socket, 'subagent.start', { subagent_id: 'child', status: 'running', goal: 'background work' })
  } else {
    emit(socket, 'message.start', {}, 'another-session')
  }

  emit(socket, 'message.complete', { text: 'waiting' })
  emit(socket, 'session.info', { running: false })
  touchBackend.mockClear()
  await advance(501)
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)
  expect(touchBackend).not.toHaveBeenCalledWith('conn:remote::writer', { activeTurn: false })
  expect(touchBackend).toHaveBeenCalledWith('conn:remote::writer', { activeTurn: true })

  // Automatic settlement removed the lease: a second release is a no-op,
  // and the same key can acquire a real, independently releasable hold again.
  touchBackend.mockClear()
  release()
  expect(touchBackend).not.toHaveBeenCalled()
  const releaseAgain = await retainGatewayForSessionTurn('remote', 'writer', 'automatic-session')
  expect(touchBackend).toHaveBeenCalledWith('conn:remote::writer', { activeTurn: true })
  releaseAgain()
  expect(touchBackend).not.toHaveBeenCalledWith('conn:remote::writer', { activeTurn: false })

  if (work === 'child') {
    emit(socket, 'subagent.complete', { subagent_id: 'child', status: 'completed' })
  } else {
    emit(socket, 'message.complete', { text: 'done' }, 'another-session')
    emit(socket, 'session.info', { running: false }, 'another-session')
  }

  expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: false })
  await advance(61 * 60_000)
  expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
})

it('explicit disposal clears activity and leases even while a child is live', async () => {
  await boot()
  const opened = openGatewayForAgent('remote', 'writer')
  await advance(1)
  await opened
  const release = await retainGatewayForSessionTurn('remote', 'writer', 'automatic-session')
  const socket = socketFor('remote')
  emit(socket, 'message.start')
  emit(socket, 'subagent.start', { subagent_id: 'child', status: 'running', goal: 'background work' })
  touchBackend.mockClear()
  act(() => closeSecondaryGateways())
  expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
  expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: false })
  expect(touchBackend).not.toHaveBeenCalledWith('conn:remote::writer', { activeTurn: true })
  touchBackend.mockClear()
  release()
  await advance(61 * 60_000)
  expect(touchBackend.mock.calls.filter(([scope]) => scope === 'conn:remote::writer')).toEqual([])
  expect(FakeWebSocket.instances).toHaveLength(2)
})

it.each(['settled turn', 'completed child'] as const)('keeps an aged route across a %s handoff', async handoff => {
  await boot()
  const opened = openGatewayForAgent('remote', 'writer')
  await advance(1)
  await opened
  const socket = socketFor('remote')

  if (handoff === 'settled turn') {
    await retainGatewayForSessionTurn('remote', 'writer', 'automatic-session')
  }

  emit(socket, 'message.start')

  if (handoff === 'completed child') {
    emit(socket, 'subagent.start', { subagent_id: 'child', status: 'running', goal: 'background work' })
    emit(socket, 'message.complete', { text: 'waiting for child' })
    emit(socket, 'session.info', { running: false })
  }

  await advance(61 * 60_000)

  if (handoff === 'settled turn') {
    emit(socket, 'session.info', { running: false })
  } else {
    emit(socket, 'subagent.complete', { subagent_id: 'child', status: 'completed' })
  }

  // Separate frames can land in different event-loop tasks.
  await advance(1)
  touchBackend.mockClear()
  emit(socket, handoff === 'settled turn' ? 'session.info' : 'message.start', { running: true })
  await advance(501)
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)
  expect(runtimeSessionOwner('automatic-session')).toEqual({ connectionId: 'remote', profile: 'writer' })
  expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: true })

  emit(socket, 'message.complete', { text: 'done' })
  emit(socket, 'session.info', { running: false })
  await advance(60_000)
  expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
  expect(touchBackend.mock.calls.filter(([scope]) => scope === 'conn:remote::writer').at(-1)?.[1]).toEqual({
    activeTurn: false
  })
})

it.each(['local', 'remote'] as const)('attributes a shared %s profile switch on the same socket', async mode => {
  const desktop = window.hermesDesktop!
  vi.mocked(desktop.getConnection).mockImplementation(async profile => ({
    ...primary,
    mode,
    connectionId: mode === 'local' ? undefined : 'primary',
    profile: profile ?? 'default',
    ...(mode === 'remote' ? { sharedRemote: true, registryScoped: true } : {}),
    ...(mode === 'local' && profile === 'writer' ? { sharedPrimary: true } : {})
  }))
  vi.mocked(desktop.getConnectionFor!).mockImplementation(async ({ connectionId, profile }) =>
    connectionId === 'primary'
      ? { ...primary, profile: profile ?? 'default', sharedRemote: true, registryScoped: true }
      : descriptor(connectionId ?? 'local', profile ?? 'default')
  )
  await boot()
  let socket = socketFor('primary')
  await act(async () => {
    if (mode === 'local') {
      await ensureGatewayProfile('writer')
    } else {
      await ensureGatewayAgent('primary', 'writer')
    }
  })
  expect($activeGatewayProfile.get()).toBe('writer')
  expect($connection.get()?.[mode === 'local' ? 'sharedPrimary' : 'sharedRemote']).toBe(true)
  expect(FakeWebSocket.instances).toHaveLength(1)
  emit(socket, 'message.start')
  expect(runtimeSessionOwner('automatic-session')).toEqual(
    mode === 'local' ? 'writer' : { connectionId: 'primary', profile: 'writer' }
  )
  act(() => socket.request())
  expect(handleServerRequest).toHaveBeenLastCalledWith(expect.objectContaining({ profile: 'writer' }))
  expect(handleServerRequest.mock.calls.at(-1)?.[0].connectionId).toBe(mode === 'local' ? undefined : 'primary')

  act(() => socket.drop())
  await advance(8_000)
  expect(socketFor('primary')).not.toBe(socket)
  socket = socketFor('primary')
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)

  const switched = ensureGatewayAgent('other', 'reader')
  await advance(1)
  await switched
  emit(socket, 'session.info', { running: true })
  expect(runtimeSessionOwner('automatic-session')).toEqual(
    mode === 'local' ? 'writer' : { connectionId: 'primary', profile: 'writer' }
  )
  act(() => socket.request())
  expect(handleServerRequest).toHaveBeenLastCalledWith(expect.objectContaining({ profile: 'writer' }))
  expect(handleServerRequest.mock.calls.at(-1)?.[0].connectionId).toBe(mode === 'local' ? undefined : 'primary')
})

it.each(['session.info', 'subagent.complete'] as const)(
  'protects each new handoff but does not renew it for repeated %s',
  async terminal => {
    const { pruneSecondaryGateways } = await import('@/store/gateway')
    await boot()
    const opened = openGatewayForAgent('remote', 'writer')
    await advance(1)
    await opened
    const socket = socketFor('remote')
    let child = 0

    const settleWork = () =>
      emit(
        socket,
        terminal,
        terminal === 'session.info' ? { running: false } : { subagent_id: `child-${child}`, status: 'completed' }
      )

    const startWork = () => {
      emit(socket, 'message.start')

      if (terminal === 'subagent.complete') {
        child++
        emit(socket, 'subagent.start', { subagent_id: `child-${child}`, status: 'running', goal: 'work' })
        emit(socket, 'message.complete', { text: 'waiting' })
        emit(socket, 'session.info', { running: false })
      }
    }

    startWork()
    await advance(61 * 60_000)

    // Each genuine cycle gets its own window on the SAME aged socket, even
    // after the previous window expired. A once-per-socket latch is wrong.
    for (let cycle = 0; cycle < 2; cycle++) {
      settleWork()
      expect(socket.readyState).toBe(FakeWebSocket.OPEN)
      await advance(1)
      startWork()
      await advance(501)
      expect(socket.readyState).toBe(FakeWebSocket.OPEN)
      expect(runtimeSessionOwner('automatic-session')).toEqual({ connectionId: 'remote', profile: 'writer' })
      expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: true })
    }

    settleWork()
    expect(socket.readyState).toBe(FakeWebSocket.OPEN)
    await advance(400)
    settleWork()
    await advance(101)
    act(() => pruneSecondaryGateways(new Set()))
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
    expect(touchBackend).toHaveBeenLastCalledWith('conn:remote::writer', { activeTurn: false })
  }
)

it.each([false, true])('minute cleanup reclaims an aged idle route with repeated notices=%s', async repeated => {
  await boot()
  const opened = openGatewayForAgent('remote', 'writer')
  await advance(1)
  await opened
  const socket = socketFor('remote')
  emit(socket, 'message.start')
  await advance(61 * 60_000)
  emit(socket, 'session.info', { running: false })
  expect(socket.readyState).toBe(FakeWebSocket.OPEN)
  touchBackend.mockClear()
  const states: number[] = []

  for (let tick = 0; tick < 3; tick++) {
    // Use the real boot interval, not a direct prune call. An idle notice
    // after expiry must not resurrect the window just before cleanup.
    await advance(60_000 - (Date.now() % 60_000) - 100)

    if (repeated) {
      emit(socket, 'session.info', { running: false })
    }

    await advance(100)
    states.push(socket.readyState)
  }

  expect(states).toEqual([FakeWebSocket.CLOSED, FakeWebSocket.CLOSED, FakeWebSocket.CLOSED])
  const activity = touchBackend.mock.calls.filter(([scope]) => scope === 'conn:remote::writer')
  expect(activity.length).toBeGreaterThan(0)
  expect(activity.every(([, options]) => options?.activeTurn === false)).toBe(true)
})
