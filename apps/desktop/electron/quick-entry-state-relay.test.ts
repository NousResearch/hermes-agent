import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createQuickEntryStateRelay, sameQuickEntryState } from './quick-entry-state-relay'

const payload = (connected: boolean, sessions: Array<{ id: string; title: string }> = []) => ({
  connected,
  sessions
})

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

function harness(overrides: Partial<Parameters<typeof createQuickEntryStateRelay>[0]> = {}) {
  const sent: unknown[] = []
  let targetAlive = true
  let latest: null | Record<string, unknown> = payload(true)

  const relay = createQuickEntryStateRelay({
    equals: sameQuickEntryState,
    isTargetAlive: () => targetAlive,
    latest: () => latest,
    send: p => void sent.push(p),
    ...overrides
  })

  return {
    get latestValue() {
      return latest
    },
    get sent() {
      return sent
    },
    killTarget() {
      targetAlive = false
    },
    relay,
    setLatest(value: null | Record<string, unknown>) {
      latest = value
    }
  }
}

describe('quick entry state relay (#95132)', () => {
  it('retries delivery until an ack arrives, then stops', () => {
    const h = harness({ retryMs: 100, maxSends: 5 })

    h.relay.deliver(payload(true))

    expect(h.sent).toHaveLength(1)

    // No ack yet → resend on the backoff.
    vi.advanceTimersByTime(100)
    expect(h.sent).toHaveLength(2)

    vi.advanceTimersByTime(100)
    expect(h.sent).toHaveLength(3)

    // A mounted composer adopts the payload and echoes it back.
    h.relay.acknowledge(payload(true))
    vi.advanceTimersByTime(500)
    expect(h.sent).toHaveLength(3)
  })

  it('stops after the retry budget instead of spinning forever behind a dead composer', () => {
    const h = harness({ retryMs: 50, maxSends: 3 })

    h.relay.deliver(payload(true))

    vi.advanceTimersByTime(50)
    vi.advanceTimersByTime(50)
    expect(h.sent).toHaveLength(3)

    // Budget exhausted: no fourth send ever.
    vi.advanceTimersByTime(10_000)
    expect(h.sent).toHaveLength(3)
  })

  it('abandons retries when the window dies mid-cycle', () => {
    const h = harness({ retryMs: 100, maxSends: 5 })

    h.relay.deliver(payload(true))
    expect(h.sent).toHaveLength(1)

    h.killTarget()
    vi.advanceTimersByTime(10_000)
    expect(h.sent).toHaveLength(1)
  })

  it('a newer cached push supersedes an in-flight replay', () => {
    const h = harness({ retryMs: 100, maxSends: 5 })
    const first = payload(false)

    h.setLatest(first)
    h.relay.deliver(first)
    expect(h.sent).toEqual([first])

    // Gateway came up while the disconnected replay was still retrying.
    const second = payload(true)
    h.setLatest(second)
    h.relay.deliver(second)

    vi.advanceTimersByTime(10_000)

    // The stale truth must NEVER be delivered again after supersession; the
    // fresh truth gets its own full bounded budget (default 5 sends).
    expect(h.sent[0]).toEqual(first)
    expect(h.sent).toHaveLength(6)

    for (const delivered of h.sent.slice(1)) {
      expect(delivered).toEqual(second)
    }
  })

  it('a duplicate deliver of the same truth does not add sends to the in-flight cycle', () => {
    const h = harness({ retryMs: 100, maxSends: 5 })
    const value = payload(true)

    h.relay.deliver(value)
    h.relay.deliver(value) // e.g. re-summon racing the first cycle

    vi.advanceTimersByTime(10_000)

    // Exactly one bounded cycle (5 sends), not two.
    expect(h.sent).toHaveLength(5)
  })

  it('a fresh deliver after a spent budget starts a clean cycle', () => {
    const h = harness({ retryMs: 10, maxSends: 2 })

    h.relay.deliver(payload(true))
    vi.advanceTimersByTime(100)
    expect(h.sent).toHaveLength(2)

    // The window reloads and the user summons again: the next request gets its
    // own full budget.
    h.relay.deliver(payload(true))
    expect(h.sent).toHaveLength(3)
    vi.advanceTimersByTime(10)
    expect(h.sent).toHaveLength(4)
  })

  it('refuses stale requests whose payload no longer matches the cache', () => {
    const h = harness({ retryMs: 100, maxSends: 5 })
    const stale = payload(false)

    h.setLatest(payload(true))

    h.relay.deliver(stale)

    vi.advanceTimersByTime(10_000)
    expect(h.sent).toHaveLength(0)
  })

  it('cancel stops everything', () => {
    const h = harness({ retryMs: 100, maxSends: 5 })

    h.relay.deliver(payload(true))
    h.relay.cancel()

    vi.advanceTimersByTime(10_000)
    expect(h.sent).toHaveLength(1)
  })
})

describe('sameQuickEntryState', () => {
  it('compares gateway truth structurally, not by reference', () => {
    expect(
      sameQuickEntryState(payload(true, [{ id: 'a', title: 'A' }]), payload(true, [{ id: 'a', title: 'A' }]))
    ).toBe(true)
    expect(sameQuickEntryState(payload(true), payload(false))).toBe(false)
    expect(
      sameQuickEntryState(payload(true, [{ id: 'a', title: 'A' }]), payload(true, [{ id: 'b', title: 'A' }]))
    ).toBe(false)
    expect(sameQuickEntryState(payload(true, []), payload(true, [{ id: 'a', title: 'A' }]))).toBe(false)
  })

  it('treats missing fields defensively (null/undefined payloads)', () => {
    expect(sameQuickEntryState(null, undefined)).toBe(true)
    expect(sameQuickEntryState(null, payload(false))).toBe(true)
    expect(sameQuickEntryState(payload(true), undefined)).toBe(false)
  })
})
