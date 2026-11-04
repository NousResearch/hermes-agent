import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import type * as GatewayStore from '@/store/gateway'
import { $liveWorkRoster } from '@/store/live-work-roster'
import type * as SessionStore from '@/store/session'
import type * as RequestRouter from '@/store/session-request-router'
import type * as SessionStates from '@/store/session-states'

import { useLiveWorkRoster } from './use-live-work-roster'

const h = vi.hoisted(() => ({
  owners: new Map<string, unknown>(),
  open: new Set<string>(),
  request: vi.fn(),
  rows: [] as { id: string }[]
}))

// Route identity as gateway.ts derives it: the launch profile in any spelling
// rides the window primary; anything else is its own scope, open or cold.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  activeGatewayProfileKey: () => 'default',
  gatewayRouteState: (connectionId: null | string, profile: string) => {
    const key = (profile || 'default') === 'default' ? 'primary:default' : `${connectionId ?? ''}::${profile}`

    return { key, open: key === 'primary:default' || h.open.has(key) }
  },
  isActivePrimary: () => true
}))

vi.mock('@/store/session', async importOriginal => ({
  ...(await importOriginal<typeof SessionStore>()),
  ownerLookupSessionRows: () => h.rows
}))

vi.mock('@/store/session-states', async importOriginal => ({
  ...(await importOriginal<typeof SessionStates>()),
  knownOwnerForSession: (id: string) => h.owners.get(id)
}))

vi.mock('@/store/session-request-router', async importOriginal => ({
  ...(await importOriginal<typeof RequestRouter>()),
  requestForSessionProfile: (
    owner: unknown,
    _ambient: unknown,
    _method: string,
    _params: unknown,
    timeoutMs?: number
  ) => h.request(owner, timeoutMs)
}))

const { $gateway } = await import('@/store/gateway')
const { $gatewayState } = await import('@/store/session')
const { $primaryGatewayRoute } = await import('@/store/gateway-primary-route')

const ownerOf = (call: unknown[]) => JSON.stringify(call[0] ?? 'ambient')
const callsFor = (owner: unknown) => h.request.mock.calls.filter(c => ownerOf(c) === JSON.stringify(owner)).length

async function tick(ms = 10_000) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  h.owners.clear()
  h.open.clear()
  h.rows = []
  h.request.mockReset()
  h.request.mockResolvedValue({ sessions: [] })
  $gateway.set({ request: vi.fn() } as unknown as HermesGateway)
  $gatewayState.set('open')
  $liveWorkRoster.set({})
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  $gateway.set(null)
  $liveWorkRoster.set({})
})

function withRows(owners: Record<string, unknown>) {
  h.rows = Object.keys(owners).map(id => ({ id }))
  Object.entries(owners).forEach(([id, owner]) => h.owners.set(id, owner))
}

describe('useLiveWorkRoster', () => {
  it('asks the launch profile once a tick however many spellings its rows carry', async () => {
    withRows({ a: 'default', b: { connectionId: 'local', profile: 'default' } })
    renderHook(() => useLiveWorkRoster())
    await tick(0)

    expect(h.request).toHaveBeenCalledTimes(1)
    await tick()
    expect(h.request).toHaveBeenCalledTimes(2)
  })

  it('gives an owner whose rows load after the connect its first read on the next tick', async () => {
    renderHook(() => useLiveWorkRoster())
    await tick(0)
    withRows({ late: 'late' })
    await tick()
    await tick()

    expect(callsFor('late')).toBe(1)
  })

  it('reads owners that only have cron or messaging rows', async () => {
    withRows({ cronRow: 'bot2' })
    renderHook(() => useLiveWorkRoster())
    await tick(0)

    expect(callsFor('bot2')).toBe(1)
  })

  it('interval ticks skip a cold idle owner and re-dial a cold busy one once a minute', async () => {
    withRows({ idle: 'idle', busy: 'busy' })
    h.request.mockImplementation(async (owner: unknown) =>
      owner === 'busy' ? { sessions: [{ session_id: 'busy', subagents: 1 }] } : { sessions: [] }
    )
    renderHook(() => useLiveWorkRoster())
    await tick(0)
    await tick(50_000)
    expect(callsFor('busy')).toBe(1)
    await tick(10_000)
    expect(callsFor('busy')).toBe(2)
    await tick(60_000)

    expect(callsFor('idle')).toBe(1)
    expect(callsFor('busy')).toBe(3)
  })

  it('a hung source never freezes the others: they keep publishing and clear when work ends', async () => {
    withRows({ hung: 'hung', live: 'live' })
    h.open.add('::hung').add('::live')
    let running = true
    h.request.mockImplementation((owner: unknown) => {
      if (owner === 'hung') {
        return new Promise(() => undefined)
      }

      return Promise.resolve({ sessions: owner === 'live' && running ? [{ session_id: 'job', background: 1 }] : [] })
    })
    renderHook(() => useLiveWorkRoster())
    await tick(0)
    expect($liveWorkRoster.get()['::live']?.sessions.job?.background).toBe(1)

    running = false
    await tick()
    expect($liveWorkRoster.get()['::live']?.sessions).toEqual({})
    // The hung read is never stacked, and it rides a one-tick timeout.
    expect(callsFor('hung')).toBe(1)
    expect(h.request.mock.calls.find(c => c[0] === 'hung')?.[1]).toBe(10_000)
  })

  it('window focus respects an active backoff and coalesces bursts', async () => {
    withRows({ flaky: 'flaky' })
    h.request.mockImplementation(async (owner: unknown) => {
      if (owner === 'flaky') {
        throw new Error('socket closed')
      }

      return { sessions: [] }
    })
    renderHook(() => useLiveWorkRoster())
    await tick(0)
    await tick(5_000)

    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    await tick(0)
    expect(callsFor('flaky')).toBe(1)
    expect(callsFor('ambient')).toBe(2)

    await tick(1_000)
    act(() => {
      window.dispatchEvent(new Event('focus'))
      window.dispatchEvent(new Event('focus'))
    })
    await tick(0)
    expect(callsFor('ambient')).toBe(2)
  })

  it('keeps retrying a source through transient failures until it recovers', async () => {
    withRows({ flaky: 'flaky' })
    h.open.add('::flaky')
    let failures = 4
    h.request.mockImplementation(async (owner: unknown) => {
      if (owner === 'flaky' && failures-- > 0) {
        throw new Error('socket closed')
      }

      return { sessions: owner === 'flaky' ? [{ session_id: 's', subagents: 1 }] : [] }
    })
    renderHook(() => useLiveWorkRoster())
    await tick(0)
    // Backoff 20 s, 40 s, 80 s, 160 s: five reads in five minutes, not thirty.
    await tick(5 * 60_000)

    expect(callsFor('flaky')).toBe(5)
    expect($liveWorkRoster.get()['::flaky']?.sessions.s?.subagents).toBe(1)
    await tick()
    expect(callsFor('flaky')).toBe(6)
  })

  it('stops for good only on a backend without the RPC', async () => {
    withRows({ old: 'old' })
    h.open.add('::old')
    h.request.mockImplementation(async (owner: unknown) => {
      if (owner === 'old') {
        throw Object.assign(new Error('method not found'), { code: -32601 })
      }

      return { sessions: [] }
    })
    renderHook(() => useLiveWorkRoster())
    await tick(0)
    await tick(10 * 60_000)

    expect(callsFor('old')).toBe(1)
  })

  it('a read from a superseded connection never prunes buckets the new one published', async () => {
    withRows({ x: 'x' })
    let finishOld: (value: unknown) => void = () => undefined
    h.request.mockImplementation((owner: unknown) =>
      owner === 'x' ? new Promise(resolve => (finishOld = resolve)) : Promise.resolve({ sessions: [] })
    )
    renderHook(() => useLiveWorkRoster())
    await tick(0)

    act(() => $gatewayState.set('closed'))
    withRows({ y: 'y' })
    h.rows = [{ id: 'y' }]
    h.request.mockImplementation(async (owner: unknown) =>
      owner === 'y' ? { sessions: [{ session_id: 'live', subagents: 1 }] } : { sessions: [] }
    )
    act(() => $gatewayState.set('open'))
    await tick(0)
    expect($liveWorkRoster.get()['::y']).toBeDefined()

    await act(async () => finishOld({ sessions: [] }))
    expect($liveWorkRoster.get()['::y']).toBeDefined()
  })

  it('starts over when the primary re-homes to another connection while the socket state stays open', async () => {
    act(() => $primaryGatewayRoute.set({ connectionId: 'old-host', mode: 'remote' }))
    h.request.mockRejectedValueOnce(Object.assign(new Error('method not found'), { code: -32601 }))
    renderHook(() => useLiveWorkRoster())
    await tick(0)
    expect(callsFor('ambient')).toBe(1)

    h.request.mockResolvedValue({ sessions: [{ session_id: 'new-host-row', subagents: 1 }] })
    act(() => $primaryGatewayRoute.set({ connectionId: 'new-host', mode: 'remote' }))
    await tick(0)

    expect(callsFor('ambient')).toBe(2)
    expect($liveWorkRoster.get()['primary:default']?.sessions['new-host-row']?.subagents).toBe(1)
  })
})
