import { describe, expect, it, vi } from 'vitest'

import {
  ENUMERATION_INITIAL_DELAY_MS,
  ENUMERATION_INTERVAL_MS,
  type EnumerationHost,
  type EnumerationTarget,
  MAX_BACKOFF_MS,
  RegistrySessionEnumerator,
  STAGGER_MS
} from './registry-session-enumerator'

interface Timer {
  at: number
  cancelled: boolean
  fire: () => void
}

/** A host with a fake clock: `advance(ms)` fires due timers in order and lets
 * each pass's fetch → remember → settle chain finish before the next. */
function harness(connections: EnumerationTarget[]) {
  let clock = 1_000_000
  const timers: Timer[] = []

  const fetchSessions = vi.fn<EnumerationHost['fetchSessions']>(async connectionId => ({
    sessions: [{ id: `${connectionId}-s1`, profile: 'default', connection_id: connectionId }]
  }))

  const remember = vi.fn<EnumerationHost['remember']>()
  const retain = vi.fn<NonNullable<EnumerationHost['retain']>>()
  let live = connections

  const host: EnumerationHost = {
    connections: () => live,
    fetchSessions,
    remember,
    retain,
    schedule: (fn, ms) => {
      const timer: Timer = { at: clock + ms, cancelled: false, fire: fn }
      timers.push(timer)

      return timer
    },
    cancel: handle => {
      ;(handle as Timer).cancelled = true
    },
    now: () => clock
  }

  const settle = () => new Promise<void>(resolve => setImmediate(resolve))

  const advance = async (ms: number) => {
    const target = clock + ms

    for (;;) {
      const due = timers.filter(t => !t.cancelled && t.at <= target).sort((a, b) => a.at - b.at)[0]

      if (!due) {
        break
      }

      due.cancelled = true
      clock = Math.max(clock, due.at)
      due.fire()
      await settle()
    }

    clock = target
    await settle()
  }

  return {
    advance,
    fetchSessions,
    host,
    remember,
    retain,
    setConnections: (next: EnumerationTarget[]) => {
      live = next
    },
    timers
  }
}

const LOCAL: EnumerationTarget = { connectionId: 'local', kind: 'local' }
const HOMELAB: EnumerationTarget = { connectionId: 'homelab', kind: 'ssh' }
const CLOUD: EnumerationTarget = { connectionId: 'cloud', kind: 'cloud' }

describe('RegistrySessionEnumerator', () => {
  it('never enumerates a single-connection registry', async () => {
    const h = harness([LOCAL])
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS + 3 * ENUMERATION_INTERVAL_MS)

    expect(h.fetchSessions).not.toHaveBeenCalled()
    expect(h.remember).not.toHaveBeenCalled()
  })

  it('holds back the first pass so it never competes with the foreground boot', async () => {
    const h = harness([LOCAL, HOMELAB])
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS - 1)
    expect(h.fetchSessions).not.toHaveBeenCalled()

    await h.advance(1)
    expect(h.fetchSessions).toHaveBeenCalledTimes(1)
    expect(h.fetchSessions).toHaveBeenCalledWith('homelab')
  })

  it('never enumerates the local connection', async () => {
    const h = harness([LOCAL, HOMELAB, CLOUD])
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS + STAGGER_MS)

    expect(h.fetchSessions.mock.calls.map(call => call[0]).sort()).toEqual(['cloud', 'homelab'])
  })

  it('never enumerates the registry primary: the primary aggregate already carries its rows', async () => {
    const h = harness([LOCAL, { ...HOMELAB, primary: true }, CLOUD])
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS + STAGGER_MS)

    expect(h.fetchSessions.mock.calls.map(call => call[0])).toEqual(['cloud'])
  })

  it('runs one pass at a time, staggered', async () => {
    const h = harness([LOCAL, HOMELAB, CLOUD])
    let concurrent = 0
    let peak = 0

    h.fetchSessions.mockImplementation(async connectionId => {
      concurrent += 1
      peak = Math.max(peak, concurrent)
      await new Promise(resolve => setImmediate(resolve))
      concurrent -= 1

      return { sessions: [{ id: 's', profile: 'default', connection_id: connectionId }] }
    })
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(1)

    await h.advance(STAGGER_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(2)
    expect(peak).toBe(1)
  })

  it('remembers an empty successful read, so a host that deleted everything clears', async () => {
    const h = harness([LOCAL, HOMELAB])
    h.fetchSessions.mockResolvedValue({ sessions: [] })
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)

    expect(h.remember).toHaveBeenCalledWith('homelab', [])
  })

  it('swallows a failed read, remembers nothing for it, and keeps the other connection', async () => {
    const h = harness([LOCAL, HOMELAB, CLOUD])
    h.fetchSessions.mockImplementation(async connectionId => {
      if (connectionId === 'homelab') {
        throw new Error('unreachable')
      }

      return { sessions: [{ id: 's', profile: 'default', connection_id: connectionId }] }
    })
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS + STAGGER_MS)

    expect(h.remember).not.toHaveBeenCalledWith('homelab', expect.anything())
    expect(h.remember).toHaveBeenCalledWith('cloud', expect.any(Array))
  })

  it('treats a null read as a failure and backs the connection off exponentially', async () => {
    const h = harness([LOCAL, HOMELAB])
    h.fetchSessions.mockResolvedValue(null)
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(1)

    // One failure → the next pass waits 2 intervals, not 1.
    await h.advance(ENUMERATION_INTERVAL_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(1)

    await h.advance(ENUMERATION_INTERVAL_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(2)

    // Two failures → 4 intervals.
    await h.advance(3 * ENUMERATION_INTERVAL_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(2)

    await h.advance(ENUMERATION_INTERVAL_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(3)
    expect(h.remember).not.toHaveBeenCalled()
  })

  it('caps the backoff so a recovered host is found again within the hour', async () => {
    const h = harness([LOCAL, HOMELAB])
    h.fetchSessions.mockResolvedValue(null)
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)

    // Many failures later the gap is MAX_BACKOFF_MS, never longer.
    await h.advance(24 * 60 * 60 * 1000)
    const callsBefore = h.fetchSessions.mock.calls.length

    await h.advance(MAX_BACKOFF_MS + ENUMERATION_INTERVAL_MS)

    expect(h.fetchSessions.mock.calls.length).toBeGreaterThan(callsBefore)
  })

  it('a success resets the backoff', async () => {
    const h = harness([LOCAL, HOMELAB])
    h.fetchSessions.mockResolvedValueOnce(null)
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)
    await h.advance(2 * ENUMERATION_INTERVAL_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(2)
    expect(h.remember).toHaveBeenCalledTimes(1)

    // Back on the plain interval.
    await h.advance(ENUMERATION_INTERVAL_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(3)
  })

  it('respects the per-connection interval', async () => {
    const h = harness([LOCAL, HOMELAB])
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)
    expect(h.fetchSessions).toHaveBeenCalledTimes(1)

    await h.advance(ENUMERATION_INTERVAL_MS - 1)
    expect(h.fetchSessions).toHaveBeenCalledTimes(1)

    await h.advance(1)
    expect(h.fetchSessions).toHaveBeenCalledTimes(2)
  })

  it('skips a connection removed between enqueue and execution', async () => {
    const h = harness([LOCAL, HOMELAB, CLOUD])
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)
    h.setConnections([LOCAL, HOMELAB])
    await h.advance(STAGGER_MS)

    expect(h.fetchSessions).toHaveBeenCalledTimes(1)
    expect(h.fetchSessions).not.toHaveBeenCalledWith('cloud')
  })

  it('tells the index which connections are still registered each cycle', async () => {
    const h = harness([LOCAL, HOMELAB])
    new RegistrySessionEnumerator(h.host).start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)

    expect([...(h.retain.mock.calls.at(-1)?.[0] ?? [])].sort()).toEqual(['homelab', 'local'])
  })

  it('stop() cancels pending timers and drops the queue', async () => {
    const h = harness([LOCAL, HOMELAB, CLOUD])
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()
    await h.advance(ENUMERATION_INITIAL_DELAY_MS)
    enumerator.stop()

    for (const timer of h.timers) {
      expect(timer.cancelled).toBe(true)
    }

    const calls = h.fetchSessions.mock.calls.length
    await h.advance(10 * ENUMERATION_INTERVAL_MS)

    expect(h.fetchSessions.mock.calls.length).toBe(calls)
  })

  it('start() is idempotent', async () => {
    const h = harness([LOCAL, HOMELAB])
    const enumerator = new RegistrySessionEnumerator(h.host)
    enumerator.start()
    enumerator.start()

    await h.advance(ENUMERATION_INITIAL_DELAY_MS)

    expect(h.fetchSessions).toHaveBeenCalledTimes(1)
  })
})
