import type { ConnectionState, GatewayEvent } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The registry calls `new HermesGateway()` internally, so mock the class with
// a controllable fake that records connect URLs and fires its state listener.
// Defined via vi.hoisted so it exists before vi.mock's factory runs (the mock
// is hoisted above normal module init).
const { FakeGateway } = vi.hoisted(() => {
  class FakeGateway {
    static instances: FakeGateway[] = []
    connectionState: ConnectionState = 'closed'
    connectUrls: string[] = []
    closed = false
    eventCbs: Array<(e: GatewayEvent) => void> = []
    requests: Array<{ method: string; params?: Record<string, unknown> }> = []
    stateCbs: Array<(s: ConnectionState) => void> = []

    constructor() {
      FakeGateway.instances.push(this)
    }

    onEvent(cb: (e: GatewayEvent) => void) {
      this.eventCbs.push(cb)

      return () => {
        this.eventCbs = this.eventCbs.filter(c => c !== cb)
      }
    }

    onState(cb: (s: ConnectionState) => void) {
      this.stateCbs.push(cb)

      return () => {
        this.stateCbs = this.stateCbs.filter(c => c !== cb)
      }
    }

    async connect(url: string) {
      this.connectUrls.push(url)
      this.setState('open')
    }

    async request<T>(method: string, params?: Record<string, unknown>): Promise<T> {
      this.requests.push({ method, params })

      return { method, ok: true, params } as T
    }

    close() {
      this.closed = true
      this.setState('closed')
    }

    emitEvent(event: GatewayEvent) {
      for (const cb of this.eventCbs) {cb(event)}
    }

    setState(state: ConnectionState) {
      this.connectionState = state

      for (const cb of this.stateCbs) {cb(state)}
    }
  }

  return { FakeGateway }
})

vi.mock('@/hermes', () => ({ HermesGateway: FakeGateway }))

import {
  activeBackendIsRemote,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForEndpoint,
  ensureGatewayForProfile,
  isRemoteBackendKey,
  pruneSecondaryGateways,
  remoteGatewayWsUrl,
  requestGatewayForEndpoint,
  setPrimaryGateway
} from './gateway'

const ENDPOINT = 'ws://192.168.1.20:8664/api/ws'

describe('remoteGatewayWsUrl', () => {
  it('returns the bare endpoint when no token', () => {
    expect(remoteGatewayWsUrl(ENDPOINT)).toBe(ENDPOINT)
  })

  it('appends ?token when none present', () => {
    expect(remoteGatewayWsUrl(ENDPOINT, 'abc')).toBe(`${ENDPOINT}?token=abc`)
  })

  it('appends &token when a query already exists', () => {
    expect(remoteGatewayWsUrl(`${ENDPOINT}?x=1`, 'a b')).toBe(`${ENDPOINT}?x=1&token=a%20b`)
  })
})

describe('isRemoteBackendKey', () => {
  it('treats ws/wss URLs as remote and profile names as local', () => {
    expect(isRemoteBackendKey(ENDPOINT)).toBe(true)
    expect(isRemoteBackendKey('wss://host/api/ws')).toBe(true)
    expect(isRemoteBackendKey('default')).toBe(false)
    expect(isRemoteBackendKey('work-profile')).toBe(false)
  })
})

describe('ensureGatewayForEndpoint', () => {
  let events: GatewayEvent[]

  beforeEach(() => {
    FakeGateway.instances = []
    events = []
    configureGatewayRegistry({ onEvent: e => events.push(e) })
    setPrimaryGateway(new FakeGateway() as never, 'default')
  })

  afterEach(() => {
    closeSecondaryGateways()
  })

  it('dials the advertised endpoint with the token and opens one socket', async () => {
    await ensureGatewayForEndpoint(ENDPOINT, 'tok1')

    // instances[0] is the primary; the remote secondary is the next one.
    const remote = FakeGateway.instances.at(-1)!
    expect(remote.connectUrls).toEqual([`${ENDPOINT}?token=tok1`])
    expect(remote.connectionState).toBe('open')
  })

  it('reuses the same socket on re-attach and refreshes the token', async () => {
    await ensureGatewayForEndpoint(ENDPOINT, 'tok1')
    const countAfterFirst = FakeGateway.instances.length

    // Drop, then re-attach with a rotated ticket.
    FakeGateway.instances.at(-1)!.setState('closed')
    await ensureGatewayForEndpoint(ENDPOINT, 'tok2')

    expect(FakeGateway.instances.length).toBe(countAfterFirst) // no new socket
    expect(FakeGateway.instances.at(-1)!.connectUrls).toEqual([`${ENDPOINT}?token=tok1`, `${ENDPOINT}?token=tok2`])
  })

  it('feeds remote gateway events into the shared event handler', async () => {
    await ensureGatewayForEndpoint(ENDPOINT, 'tok1')
    const remote = FakeGateway.instances.at(-1)!

    remote.emitEvent({ type: 'message.delta', session_id: 's1', payload: { text: 'hi' } } as GatewayEvent)

    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'message.delta', session_id: 's1' })
  })

  it('can request a remote gateway in the background without making it active', async () => {
    await ensureGatewayForProfile('default')
    expect(activeBackendIsRemote()).toBe(false)

    await expect(requestGatewayForEndpoint(ENDPOINT, 'cloud.ping', { session_id: 's1' }, 'tok1')).resolves.toEqual({
      method: 'cloud.ping',
      ok: true,
      params: { session_id: 's1' }
    })

    const remote = FakeGateway.instances.at(-1)!
    expect(remote.connectUrls).toEqual([`${ENDPOINT}?token=tok1`])
    expect(remote.requests).toEqual([{ method: 'cloud.ping', params: { session_id: 's1' } }])
    expect(activeBackendIsRemote()).toBe(false)
  })

  it('is pruned and closed when neither active nor kept', async () => {
    await ensureGatewayForEndpoint(ENDPOINT, 'tok1')
    const remote = FakeGateway.instances.at(-1)!

    // Make the primary active again, then prune with an empty keep set.
    pruneSecondaryGateways(new Set(['default']))

    // The remote was active, so it survives a prune that doesn't target it...
    expect(remote.closed).toBe(false)
  })

  it('closes a backgrounded remote when it is not in the keep set', async () => {
    await ensureGatewayForEndpoint(ENDPOINT, 'tok1')
    const remote = FakeGateway.instances.at(-1)!

    // Switch active away to a different remote so ENDPOINT is backgrounded.
    const other = 'ws://10.0.0.9:8664/api/ws'
    await ensureGatewayForEndpoint(other, 'tok2')

    pruneSecondaryGateways(new Set([other]))

    expect(remote.closed).toBe(true)
  })
})

describe('activeBackendIsRemote', () => {
  beforeEach(() => {
    FakeGateway.instances = []
    configureGatewayRegistry({ onEvent: () => undefined })
    setPrimaryGateway(new FakeGateway() as never, 'default')
  })

  afterEach(() => {
    closeSecondaryGateways()
  })

  it('flips with the active backend', async () => {
    const { activeBackendIsRemote, ensureGatewayForProfile } = await import('./gateway')

    await ensureGatewayForProfile('default')
    expect(activeBackendIsRemote()).toBe(false)

    await ensureGatewayForEndpoint(ENDPOINT, 'tok')
    expect(activeBackendIsRemote()).toBe(true)

    await ensureGatewayForProfile('default')
    expect(activeBackendIsRemote()).toBe(false)
  })
})
