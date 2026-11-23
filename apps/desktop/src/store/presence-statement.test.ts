// A report is a claim that the user is here AND this window can show them the reply. Whatever the
// reporter last told a connection must be corrected the moment it stops being true; old samples
// and clock changes never make it claim more than it knows.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { PresenceSnapshot } from '../../electron/presence'

import { createPresenceReporter, type PresenceChannel, type PresenceView } from './presence'

function fakeChannel(): PresenceChannel & { calls: Record<string, unknown>[] } {
  const calls: Record<string, unknown>[] = []

  return {
    calls,
    clientActivity: true,
    connectionState: 'open',
    request: async <T>(method: string, params: Record<string, unknown>) => {
      if (method === 'client.activity') {
        calls.push(params)
      }

      return {} as T
    }
  }
}

let seq = 0

const snap = (over: Partial<PresenceSnapshot> = {}): PresenceSnapshot => ({
  idleSeconds: 1,
  locked: false,
  sampledAt: Date.now(),
  seq: ++seq,
  suspended: false,
  system: true,
  ...over
})

function harness({ system }: { system?: boolean } = {}) {
  const state = {
    canAlert: true,
    focused: true,
    view: null as null | PresenceView,
    visible: true
  }

  const a = fakeChannel()
  state.view = { channel: a, kind: 'session', sessionId: 'A' }
  const target = new EventTarget()
  let push: (next: PresenceSnapshot) => void = () => undefined
  let reply: (next: PresenceSnapshot) => void = () => undefined
  let current = snap()

  const reporter = createPresenceReporter({
    canAlert: () => state.canAlert,
    focused: () => state.focused,
    inputTarget: target,
    system: system
      ? {
          get: () =>
            new Promise<PresenceSnapshot>(resolve => {
              reply = resolve
            }),
          onChange: fn => {
            push = fn

            return () => undefined
          }
        }
      : undefined,
    view: () => state.view!,
    visible: () => state.visible
  })

  return {
    a,
    reporter,
    state,
    input: () => target.dispatchEvent(new Event('keydown')),
    push: (over: Partial<PresenceSnapshot> = {}) => push((current = snap(over))),
    /** Answer the newest outstanding `get()` (by default with the newest sample). */
    reply: (value: PresenceSnapshot = current) => reply(value),
    settle: (ms = 0) => vi.advanceTimersByTimeAsync(ms)
  }
}

beforeEach(() => void vi.useFakeTimers())
afterEach(() => void vi.useRealTimers())

describe('the statement follows the session in view', () => {
  it('A then B on one connection: the second report lists only B', async () => {
    const h = harness()

    h.input()
    await h.settle()
    h.state.view = { channel: h.a, kind: 'session', sessionId: 'B' }
    h.reporter.refresh()
    await h.settle()

    expect(h.a.calls.map(call => call.sessions)).toEqual([['A'], ['B']])
  })

  it('withdraws at once when nothing is in view any more', async () => {
    const h = harness()

    h.input()
    await h.settle()
    h.state.view = { channel: h.a, kind: 'none' }
    h.reporter.refresh()
    await h.settle()

    expect(h.a.calls.at(-1)).toMatchObject({ sessions: [] })
  })

  it('withdraws from the old connection and claims on the new one when the owner moves', async () => {
    const h = harness()
    const b = fakeChannel()

    h.input()
    await h.settle()
    h.state.view = { channel: b, kind: 'session', sessionId: 'A' }
    h.reporter.refresh()
    await h.settle()

    expect(h.a.calls.map(call => call.sessions)).toEqual([['A'], []])
    expect(b.calls.map(call => call.sessions)).toEqual([['A']])
  })

  it('withdraws when the session cannot be routed, and claims nothing on it', async () => {
    const h = harness()

    h.input()
    await h.settle()
    h.state.view = { kind: 'unroutable' }
    h.input()
    h.reporter.refresh()
    await h.settle(60_000)

    expect(h.a.calls.map(call => call.sessions)).toEqual([['A'], []])
  })
})

describe('can_alert is withdrawn when it stops being true', () => {
  it('page tier, visible to hidden directly with notifications off: an empty list, never true again', async () => {
    const h = harness()

    h.input()
    await h.settle()
    h.state.visible = false
    h.state.canAlert = false
    h.reporter.refresh()
    await h.settle(120_000)

    expect(h.a.calls).toEqual([
      expect.objectContaining({ can_alert: true, sessions: ['A'] }),
      expect.objectContaining({ sessions: [] })
    ])
  })

  it('system tier, visible to hidden directly: can_alert false with the real idle time, never true again', async () => {
    const h = harness({ system: true })

    h.reply(snap({ idleSeconds: 3 }))
    await h.settle()
    h.state.visible = false
    h.state.canAlert = false
    h.push({ idleSeconds: 7 })
    await h.settle()
    h.push({ idleSeconds: 0 })
    await h.settle()

    expect(h.a.calls.map(call => [call.can_alert, call.idle_seconds])).toEqual([
      [true, 3],
      [false, 7]
    ])
  })

  it('stale evidence and only can_alert turning false: an empty list, not a fresh-looking claim', async () => {
    const h = harness({ system: true })

    h.reply(snap({ idleSeconds: 3 }))
    await h.settle()
    h.push({ idleSeconds: 90 })
    await h.settle()
    h.state.canAlert = false
    h.push({ idleSeconds: 95 })
    await h.settle()

    expect(h.a.calls).toEqual([expect.objectContaining({ sessions: ['A'] }), expect.objectContaining({ sessions: [] })])
  })

  it('sends nothing when evidence merely goes stale', async () => {
    const h = harness()

    h.input()
    await h.settle(60_000)
    const claims = h.a.calls.length
    await h.settle(300_000)

    expect(h.a.calls).toHaveLength(claims)
    expect(h.a.calls.every(call => (call.sessions as string[])[0] === 'A')).toBe(true)
  })
})

describe('lock and suspend always end the claim', () => {
  it('sends away on the owning connection even before any report of its own', async () => {
    const h = harness({ system: true })

    h.reply(snap({ idleSeconds: 61 }))
    await h.settle()
    h.push({ idleSeconds: 62, locked: true })
    await h.settle()

    expect(h.a.calls).toEqual([expect.objectContaining({ away: true, sessions: [] })])
  })

  it('sends away once per episode on every connection holding a statement', async () => {
    const h = harness({ system: true })
    const b = fakeChannel()

    h.reply(snap({ idleSeconds: 1 }))
    await h.settle()
    h.state.view = { channel: b, kind: 'session', sessionId: 'A' }
    h.push({ idleSeconds: 2 })
    await h.settle()
    h.push({ locked: true })
    h.push({ locked: true, suspended: true })
    await h.settle(60_000)

    expect(h.a.calls.filter(call => call.away)).toHaveLength(1)
    expect(b.calls.filter(call => call.away)).toHaveLength(1)
  })

  it('a late get() reply sampled before the lock cannot undo the lock push', async () => {
    const h = harness({ system: true })
    const beforeLock = snap({ idleSeconds: 1 })

    h.push({ idleSeconds: 1 })
    await h.settle()
    h.push({ locked: true })
    await h.settle()
    h.reply(beforeLock)
    await h.settle()

    expect(h.a.calls.at(-1)).toMatchObject({ away: true })
    expect(h.reporter.evidence().active).toBe(false)
  })
})

describe('elapsed time is monotonic', () => {
  it('a 90-second wall-clock rollback does not turn old page input fresh', async () => {
    const h = harness()

    h.input()
    await h.settle(90_000)
    vi.setSystemTime(Date.now() - 90_000)

    expect(h.reporter.evidence().active).toBe(false)
  })

  it('a late delivery counts the time since main sampled it', async () => {
    const h = harness({ system: true })

    h.reply(snap({ idleSeconds: 40, sampledAt: Date.now() - 30_000 }))
    await h.settle()

    expect(h.reporter.evidence()).toMatchObject({ active: false, idleSeconds: 70 })
    expect(h.a.calls).toEqual([])
  })

  it('drops held page input on resume from suspend', async () => {
    const h = harness({ system: true })

    h.reply(snap({ idleSeconds: null, system: false }))
    await h.settle()
    h.input()
    await h.settle()
    h.push({ idleSeconds: null, suspended: true, system: false })
    h.push({ idleSeconds: null, system: false })
    await h.settle()

    expect(h.reporter.evidence()).toMatchObject({ active: false, idleSeconds: null })
    expect(h.a.calls.filter(call => (call.sessions as string[]).length > 0)).toHaveLength(1)
  })
})
