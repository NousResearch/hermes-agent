import { type GatewayEvent, type GatewayWsUrlResult } from '@hermes/shared'
import { act, cleanup, render } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/api/client'
import { useBackgroundQueueDrain } from '@/app/session/hooks/use-background-queue-drain'
import type { HermesConnection } from '@/global'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
import {
  activeGateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  disposeSecondariesForConnection,
  liveSecondaryConnectionIds,
  pruneSecondaryGateways,
  reconnectSecondaryGateways,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} from '@/store/gateway'
import { requestForSessionProfile } from '@/store/session-request-router'
import {
  $sessionStates,
  $sessionTiles,
  $workingSessionIds,
  clearAllSessionStates,
  liveSessionScopes,
  publishSessionState,
  recordSessionEventScope
} from '@/store/session-states'

// Real request router, registry and JSON-RPC client; only the browser network
// boundary and Electron descriptor lookup are simulated. No backend/LLM calls.
const owner = { connectionId: 'h2-remote', mode: 'remote' as const, profile: 'research' }
const sessionId = 'h2-running-session'
const remoteUrl = 'wss://h2-remote.invalid/api/ws?profile=research'
const sockets: NetworkSocket[] = []
interface SequencedEvent extends GatewayEvent {
  seq: number
}

const history: SequencedEvent[] = []
const onEvent = vi.fn<(event: GatewayEvent) => void>()
const onActiveRouteChanged = vi.fn()
const ambient = vi.fn<() => Promise<never>>()
let primary: HermesGateway
let holdReplay = false
const pendingReplays: Array<() => void> = []

class NetworkSocket extends EventTarget {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  readyState = NetworkSocket.CONNECTING
  readonly requests: Array<{ id: number; method: string; params: Record<string, unknown> }> = []

  constructor(readonly url: string) {
    super()
    sockets.push(this)
    setTimeout(() => {
      if (this.readyState !== NetworkSocket.CONNECTING) {
        return
      }

      this.readyState = NetworkSocket.OPEN
      this.dispatchEvent(new Event('open'))
    }, 1)
  }

  send(data: string): void {
    if (this.readyState !== NetworkSocket.OPEN) {
      throw new Error('send on closed mock socket')
    }

    const request = JSON.parse(data) as (typeof this.requests)[number]
    this.requests.push(request)

    const result =
      request.method === 'session.events.since'
        ? {
            events: history.filter(
              event => event.session_id === request.params.session_id && event.seq > Number(request.params.last_seen)
            )
          }
        : { status: 'streaming' }

    // Server ACK is asynchronous, after request() installed its pending entry.
    const reply = () => this.receive({ jsonrpc: '2.0', id: request.id, result })

    if (holdReplay && request.method === 'session.events.since') {
      pendingReplays.push(reply)
    } else {
      setTimeout(reply, 1)
    }
  }

  receive(frame: unknown): void {
    if (this.readyState !== NetworkSocket.OPEN) {
      return
    }

    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }

  close(): void {
    if (this.readyState >= NetworkSocket.CLOSING) {
      return
    }

    this.readyState = NetworkSocket.CLOSING
    setTimeout(() => this.finishClose(1000), 1)
  }

  networkDrop(withError = false): void {
    // A network-originated close is an asynchronous browser event, not a
    // synchronous call to the registry's close/dispose machinery.
    setTimeout(() => {
      if (withError) {
        this.dispatchEvent(new Event('error'))
      }

      this.finishClose(1006)
    }, 1)
  }

  private finishClose(code: number): void {
    if (this.readyState === NetworkSocket.CLOSED) {
      return
    }

    this.readyState = NetworkSocket.CLOSED
    this.dispatchEvent(new CloseEvent('close', { code, wasClean: code === 1000 }))
  }
}

function publish(type: GatewayEvent['type'], payload: Record<string, unknown>): GatewayEvent {
  const event: SequencedEvent = { type, session_id: sessionId, seq: history.length + 1, payload }
  history.push(event)

  for (const socket of sockets.filter(socket => socket.url === remoteUrl)) {
    socket.receive({ jsonrpc: '2.0', method: 'event', params: event })
  }

  return event
}

beforeEach(async () => {
  vi.useFakeTimers()
  vi.spyOn(Math, 'random').mockReturnValue(0.5)
  vi.stubGlobal('WebSocket', NetworkSocket)
  sockets.length = 0
  history.length = 0
  holdReplay = false
  pendingReplays.length = 0
  vi.clearAllMocks()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnectionFor: vi.fn(async () => ({
      ...owner,
      authMode: 'token',
      token: 'test-only',
      wsUrl: remoteUrl,
      baseUrl: 'https://h2-remote.invalid'
    })),
    getGatewayWsUrlFor: vi.fn(async () => remoteUrl)
  }
  configureGatewayRegistry({
    activeConnectionId: () => 'local',
    foregroundScopes: () => new Set(),
    onEvent,
    onActiveRouteChanged
  })
  primary = new HermesGateway()
  const connected = primary.connect('wss://h2-local.invalid/api/ws')
  await vi.advanceTimersByTimeAsync(1)
  await connected
  setPrimaryGateway(primary, 'default')
  setPrimaryGatewayConnectionId('local')
  onActiveRouteChanged.mockClear()
})

afterEach(() => {
  closeSecondaryGateways()
  primary.close()
  setPrimaryGateway(null)
  setPrimaryGatewayConnectionId(null)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

const submit = () =>
  requestForSessionProfile(owner, ambient, 'prompt.submit', { session_id: sessionId, text: 'review fixture' })

async function submittedTurn() {
  const pending = submit()
  await vi.advanceTimersByTimeAsync(10)
  await pending
  publish('message.start', {})
}

afterEach(() => {
  clearAllSessionStates()
  $sessionTiles.set([])
})

it.each(['queue-first', 'idle-first'] as const)(
  'mounted queue waits for authoritative idle catch-up: %s',
  async order => {
    const trace: Array<Record<string, unknown>> = []
    onEvent.mockImplementation(event => {
      trace.push({ edge: event.type, at: Date.now() })
      recordSessionEventScope(event)

      if (event.type === 'message.start') {
        publishSessionState(sessionId, { ...createClientSessionState(sessionId), busy: true })
      }

      if (event.type === 'session.info' && (event.payload as { running?: boolean })?.running === false) {
        publishSessionState(sessionId, createClientSessionState(sessionId))
      }
    })

    const drainSubmit = vi.fn(async (text: string) => {
      trace.push({ edge: 'queue-acquire', at: Date.now() })
      await requestForSessionProfile(owner, ambient, 'prompt.submit', { session_id: sessionId, text })
      trace.push({ edge: 'queue-ack', at: Date.now() })

      return true
    })

    const offWorking = $workingSessionIds.listen(() =>
      pruneSecondaryGateways(liveSessionScopes(), { preserveTurnLeases: true })
    )

    const runtimeMap = { current: new Map([[sessionId, sessionId]]) }

    function QueueProbe() {
      useBackgroundQueueDrain({
        enabled: true,
        runtimeIdByStoredSessionIdRef: runtimeMap,
        selectedStoredSessionId: 'other',
        submitText: drainSubmit
      })

      return null
    }

    try {
      await submittedTurn()
      render(createElement(QueueProbe))
      holdReplay = true

      if (order === 'queue-first') {
        await act(async () => {
          enqueueQueuedPrompt(sessionId, { text: 'queued continuation', attachments: [] })
        })
      }

      expect(drainSubmit).not.toHaveBeenCalled()
      await act(async () => {
        sockets.find(s => s.url === remoteUrl)!.networkDrop()
        await vi.advanceTimersByTimeAsync(1)
        publish('session.info', { running: false })
        // Isolate this paired boundary from autonomous-backoff latency; F1's
        // separate probe covers the unassisted reconnect/prune consumer.
        reconnectSecondaryGateways()
        await vi.dynamicImportSettled()
        await vi.advanceTimersByTimeAsync(5)
      })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000)
      })
      expect(drainSubmit).not.toHaveBeenCalled()
      expect(pendingReplays).toHaveLength(1)
      await act(async () => {
        pendingReplays.shift()!()
      })

      if (order === 'idle-first') {
        await act(async () => {
          enqueueQueuedPrompt(sessionId, { text: 'queued continuation', attachments: [] })
        })
      }

      await act(async () => {
        await vi.advanceTimersByTimeAsync(10)
      })
      expect(drainSubmit).toHaveBeenCalledExactlyOnceWith(
        'queued continuation',
        expect.objectContaining({ fromQueue: true, sessionId, storedSessionId: sessionId })
      )
      expect(getQueuedPrompts(sessionId)).toHaveLength(0)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(501)
      })
      console.log(
        'PAIRED_BOUNDARY_TRACE',
        JSON.stringify({ trace, live: [...liveSecondaryConnectionIds()], socketStates: sockets.map(s => s.readyState) })
      )
      expect(liveSecondaryConnectionIds()).toEqual(new Set([owner.connectionId]))
      expect(sockets.filter(s => s.url === remoteUrl)[1].readyState).toBe(NetworkSocket.OPEN)
      expect(sockets.flatMap(s => s.requests).filter(r => r.method === 'prompt.submit')).toHaveLength(2)
      await act(async () => {
        publish('message.start', {})
        publish('session.info', { running: false })
        await vi.advanceTimersByTimeAsync(501)
      })
      expect(liveSecondaryConnectionIds()).toEqual(new Set())
    } finally {
      offWorking()
      cleanup()
      $queuedPromptsBySession.set({})
      onEvent.mockReset()
    }
  }
)

it('routed acquisition waits behind replay before canceling old idle release', async () => {
  await submittedTurn()
  holdReplay = true
  sockets.find(s => s.url === remoteUrl)!.networkDrop()
  await vi.advanceTimersByTimeAsync(1)
  publish('session.info', { running: false })
  reconnectSecondaryGateways()
  await vi.dynamicImportSettled()
  await vi.advanceTimersByTimeAsync(10)
  const next = submit()
  await vi.advanceTimersByTimeAsync(1000)
  expect(sockets.flatMap(s => s.requests).filter(r => r.method === 'prompt.submit')).toHaveLength(1)
  pendingReplays.shift()!()
  await vi.advanceTimersByTimeAsync(10)
  await next
  await vi.advanceTimersByTimeAsync(1000)
  expect(sockets.filter(s => s.url === remoteUrl)[1].readyState).toBe(NetworkSocket.OPEN)
  expect(sockets.flatMap(s => s.requests).filter(r => r.method === 'prompt.submit')).toHaveLength(2)
  publish('message.start', {})
  publish('session.info', { running: false })
  await vi.advanceTimersByTimeAsync(501)
  expect(liveSecondaryConnectionIds()).toEqual(new Set())
})

it('replay timeout preserves held work until reclamation and rejects early submission', async () => {
  await submittedTurn()
  recordSessionEventScope({ ...owner, session_id: sessionId })
  publishSessionState(sessionId, { ...createClientSessionState(sessionId), busy: true })
  holdReplay = true
  sockets.find(s => s.url === remoteUrl)!.networkDrop()
  await vi.advanceTimersByTimeAsync(1000)
  await vi.dynamicImportSettled()
  await vi.advanceTimersByTimeAsync(10)

  const outcome = submit().then(
    () => 'submitted',
    () => 'not-ready'
  )

  await vi.advanceTimersByTimeAsync(10001)
  expect(await outcome).toBe('not-ready')
  expect(sockets.flatMap(s => s.requests).filter(r => r.method === 'prompt.submit')).toHaveLength(1)
  expect($sessionStates.get()[sessionId].busy).toBe(true)
  expect(liveSecondaryConnectionIds()).toEqual(new Set([owner.connectionId]))
  publish('session.reclaimed', {})
  await vi.advanceTimersByTimeAsync(60000)
  expect(liveSecondaryConnectionIds()).toEqual(new Set())
  expect(sockets.filter(s => s.url === remoteUrl)).toHaveLength(2)
})

describe('independent delayed reconnect controls', () => {
  it('idle-first queue acquisition cancels the settlement timer control', async () => {
    await submittedTurn()
    publish('session.info', { running: false })
    const followup = submit()
    await vi.advanceTimersByTimeAsync(10)
    await followup
    await vi.advanceTimersByTimeAsync(501)
    expect(liveSecondaryConnectionIds()).toEqual(new Set([owner.connectionId]))
    expect(sockets.find(s => s.url === remoteUrl)!.readyState).toBe(NetworkSocket.OPEN)
    publish('message.start', {})
    publish('session.info', { running: false })
    await vi.advanceTimersByTimeAsync(501)
    expect(liveSecondaryConnectionIds()).toEqual(new Set())
  })

  it('current sole-turn reconnect survives the live-work pruning subscription', async () => {
    await submittedTurn()
    recordSessionEventScope({ ...owner, session_id: sessionId })
    publishSessionState(sessionId, { ...createClientSessionState(sessionId), busy: true, awaitingResponse: true })
    recordSessionEventScope({ ...owner, session_id: 'stale-unheld' })
    publishSessionState('stale-unheld', {
      ...createClientSessionState('stale-unheld'),
      busy: true,
      awaitingResponse: true
    })
    // Same public derived-store edge and registry pruner as use-gateway-boot;
    // fixture has no foreground tiles/owner holds or other live scopes.
    const off = $workingSessionIds.listen(() => pruneSecondaryGateways(liveSessionScopes()))

    try {
      pruneSecondaryGateways(liveSessionScopes())
      const original = sockets.find(s => s.url === remoteUrl)!
      expect(original.readyState).toBe(NetworkSocket.OPEN)
      original.networkDrop()
      await vi.advanceTimersByTimeAsync(1)
      publish('message.delta', { text: 'during outage' })
      onEvent.mockClear()
      await vi.advanceTimersByTimeAsync(1000)
      await vi.dynamicImportSettled()
      await vi.advanceTimersByTimeAsync(50)
      expect($sessionStates.get()['stale-unheld'].busy).toBe(false)
      expect($sessionStates.get()['stale-unheld'].awaitingResponse).toBe(false)
      expect
        .soft(liveSecondaryConnectionIds(), 'running background route survives reconnect')
        .toEqual(new Set([owner.connectionId]))
      expect
        .soft(
          sockets.filter(s => s.url === remoteUrl).some(s => s.readyState === NetworkSocket.OPEN),
          'reconnected socket remains open'
        )
        .toBe(true)
      expect
        .soft(onEvent, 'outage event reaches fan-in')
        .toHaveBeenCalledWith(expect.objectContaining({ type: 'message.delta', session_id: sessionId }))
      publish('message.complete', { text: 'completed after reconnect' })
      expect
        .soft(onEvent, 'completion reaches fan-in')
        .toHaveBeenCalledWith(expect.objectContaining({ type: 'message.complete', session_id: sessionId }))
    } finally {
      off()
    }
  })

  it.each(['resolve', 'reject'] as const)('old reconnect %s after removal leaves replacement leased', async outcome => {
    await submittedTurn()
    let resolve!: (value: HermesConnection) => void
    let reject!: (error: Error) => void

    const held = new Promise<HermesConnection>((yes, no) => {
      resolve = yes
      reject = no
    })

    const lookup = vi.mocked(window.hermesDesktop!.getConnectionFor!)
    const normal = lookup.getMockImplementation()!
    lookup.mockImplementationOnce(() => held)
    sockets.find(s => s.url === remoteUrl)!.networkDrop()
    await vi.advanceTimersByTimeAsync(1000)
    await vi.dynamicImportSettled()
    await vi.advanceTimersByTimeAsync(10)
    expect(lookup).toHaveBeenCalledTimes(2)
    disposeSecondariesForConnection(owner.connectionId)
    await submittedTurn()
    const replacement = sockets.filter(s => s.url === remoteUrl)[1]
    onEvent.mockClear()

    if (outcome === 'resolve') {
      resolve(await normal({ connectionId: owner.connectionId, profile: owner.profile }))
    } else {
      reject(new Error(`No connection with id "${owner.connectionId}"`))
    }

    await vi.advanceTimersByTimeAsync(50)
    expect(replacement.readyState).toBe(NetworkSocket.OPEN)
    expect(liveSecondaryConnectionIds()).toEqual(new Set([owner.connectionId]))
    expect(onEvent).not.toHaveBeenCalled()
    expect(activeGateway()).toBe(primary)
    publish('session.info', { running: false })
    await vi.advanceTimersByTimeAsync(501)
    expect(replacement.readyState).toBe(NetworkSocket.CLOSED)
    expect(liveSecondaryConnectionIds()).toEqual(new Set())
    const total = sockets.length
    await vi.advanceTimersByTimeAsync(60000)
    expect(sockets).toHaveLength(total)
  })

  it.each(['close', 'remove'] as const)(
    'explicit %s cancels a pending material-edit redial before releasing the turn',
    async operation => {
      await submittedTurn()
      disposeSecondariesForConnection(owner.connectionId, { redial: true })

      if (operation === 'close') {
        closeSecondaryGateways()
      } else {
        disposeSecondariesForConnection(owner.connectionId)
      }

      await vi.advanceTimersByTimeAsync(60000)
      expect(sockets.filter(s => s.url === remoteUrl)).toHaveLength(1)
      expect(liveSecondaryConnectionIds()).toEqual(new Set())
      await submittedTurn()
      publish('session.info', { running: false })
      await vi.advanceTimersByTimeAsync(501)
      expect(liveSecondaryConnectionIds()).toEqual(new Set())
    }
  )

  it('material edit waits for sole-turn settlement then redials once', async () => {
    await submittedTurn()
    const first = sockets.find(s => s.url === remoteUrl)!
    disposeSecondariesForConnection(owner.connectionId, { redial: true })
    await vi.advanceTimersByTimeAsync(1000)
    expect(first.readyState).toBe(NetworkSocket.OPEN)
    expect(sockets.filter(s => s.url === remoteUrl)).toHaveLength(1)
    publish('session.info', { running: false })
    await vi.advanceTimersByTimeAsync(510)
    expect(first.readyState).toBe(NetworkSocket.CLOSED)
    expect(sockets.filter(s => s.url === remoteUrl)).toHaveLength(2)
    expect(sockets.filter(s => s.url === remoteUrl)[1].readyState).toBe(NetworkSocket.OPEN)
    disposeSecondariesForConnection(owner.connectionId)
    await vi.advanceTimersByTimeAsync(60000)
    expect(sockets.filter(s => s.url === remoteUrl)).toHaveLength(2)
  })

  it.each(['resolve', 'reject', 'timeout', 'url-resolve'] as const)(
    'late disposed reconnect %s must not retire replacement busy state',
    async outcome => {
      await submittedTurn()
      let resolve!: (value: HermesConnection | GatewayWsUrlResult) => void
      let reject!: (error: Error) => void

      const held = new Promise<HermesConnection | GatewayWsUrlResult>((yes, no) => {
        resolve = yes
        reject = no
      })

      const lookup =
        outcome === 'url-resolve'
          ? vi.mocked(window.hermesDesktop!.getGatewayWsUrlFor!)
          : vi.mocked(window.hermesDesktop!.getConnectionFor!)

      const normal = lookup.getMockImplementation()!

      if (outcome === 'url-resolve') {
        vi.mocked(window.hermesDesktop!.getGatewayWsUrlFor!).mockImplementationOnce(() => held as Promise<GatewayWsUrlResult>)
      } else {
        vi.mocked(window.hermesDesktop!.getConnectionFor!).mockImplementationOnce(
          () => held as Promise<HermesConnection>
        )
      }

      sockets.find(s => s.url === remoteUrl)!.networkDrop()
      await vi.advanceTimersByTimeAsync(1000)
      await vi.dynamicImportSettled()
      await vi.advanceTimersByTimeAsync(10)
      expect(lookup).toHaveBeenCalledTimes(2)
      disposeSecondariesForConnection(owner.connectionId)
      await submittedTurn()
      $sessionTiles.set([{ runtimeId: sessionId, storedSessionId: 'replacement-tile', ownerRoute: owner }])
      // Seed the same scoped busy cache a new live message.start publishes.
      recordSessionEventScope({ ...owner, session_id: sessionId })
      publishSessionState(sessionId, { ...createClientSessionState(sessionId), busy: true, awaitingResponse: true })
      expect($sessionStates.get()[sessionId].busy).toBe(true)
      const replacement = sockets.filter(s => s.url === remoteUrl)[1]
      const scope = 'conn:h2-remote::research'
      expect(liveSessionScopes()).toEqual(new Set([scope]))
      pruneSecondaryGateways(liveSessionScopes())
      expect(replacement.readyState).toBe(NetworkSocket.OPEN)

      if (outcome === 'reject') {
        reject(new Error(`No connection with id "${owner.connectionId}"`))
      } else if (outcome === 'timeout') {
        await vi.advanceTimersByTimeAsync(20000)
        resolve(await normal({ connectionId: owner.connectionId, profile: owner.profile }))
      } else {
        resolve(await normal({ connectionId: owner.connectionId, profile: owner.profile }))
      }

      await vi.advanceTimersByTimeAsync(50)
      expect(sockets.filter(s => s.url === remoteUrl)).toHaveLength(2)
      expect($sessionTiles.get()[0].runtimeId).toBe(sessionId)
      expect.soft($sessionStates.get()[sessionId].busy, 'replacement busy survives stale completion').toBe(true)
      expect
        .soft($sessionStates.get()[sessionId].awaitingResponse, 'replacement awaiting survives stale completion')
        .toBe(true)
      expect.soft(liveSessionScopes(), 'replacement is still live work').toEqual(new Set([scope]))
      pruneSecondaryGateways(liveSessionScopes())
      await vi.advanceTimersByTimeAsync(2)
      expect
        .soft(replacement.readyState, 'normal live-work pruning preserves replacement socket')
        .toBe(NetworkSocket.OPEN)
      expect
        .soft(liveSecondaryConnectionIds(), 'replacement route remains registered')
        .toEqual(new Set([owner.connectionId]))
    }
  )

  it.each(['connection', 'profile'] as const)(
    'authoritative settlement isolates %s for the same session id',
    async dimension => {
      vi.mocked(window.hermesDesktop!.getConnectionFor!).mockImplementation(
        async ({ connectionId, profile }) =>
          ({
            connectionId,
            profile,
            mode: 'remote',
            authMode: 'token',
            token: 'test-only',
            baseUrl: `https://${connectionId}.invalid`,
            wsUrl: `wss://${connectionId}.invalid/api/ws?profile=${profile}`
          }) as HermesConnection
      )
      vi.mocked(window.hermesDesktop!.getGatewayWsUrlFor!).mockImplementation(
        async ({ connectionId, profile }) => `wss://${connectionId}.invalid/api/ws?profile=${profile}`
      )
      const first = requestForSessionProfile(owner, ambient, 'prompt.submit', { session_id: sessionId, text: 'first' })

      const otherOwner = {
        ...owner,
        ...(dimension === 'connection' ? { connectionId: 'review-other' } : { profile: 'other' })
      }

      const second = requestForSessionProfile(otherOwner, ambient, 'prompt.submit', {
        session_id: sessionId,
        text: 'second'
      })

      await vi.advanceTimersByTimeAsync(10)
      await Promise.all([first, second])
      const a = sockets.find(s => s.url === `wss://${owner.connectionId}.invalid/api/ws?profile=${owner.profile}`)!

      const b = sockets.find(
        s => s.url === `wss://${otherOwner.connectionId}.invalid/api/ws?profile=${otherOwner.profile}`
      )!

      a.receive({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'session.info', session_id: sessionId, payload: { running: false } }
      })
      await vi.advanceTimersByTimeAsync(501)
      expect(a.readyState).toBe(NetworkSocket.CLOSED)
      expect(b.readyState).toBe(NetworkSocket.OPEN)
      expect(liveSecondaryConnectionIds()).toEqual(new Set([otherOwner.connectionId]))
      b.receive({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'session.info', session_id: sessionId, payload: { running: false } }
      })
      await vi.advanceTimersByTimeAsync(501)
      expect(b.readyState).toBe(NetworkSocket.CLOSED)
      expect(liveSecondaryConnectionIds()).toEqual(new Set())
    }
  )
})
