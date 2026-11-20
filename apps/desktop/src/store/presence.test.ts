import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import type { PresenceSnapshot } from '../../electron/presence'

import { $browserNotificationPermission } from './browser-notification-permission'
import { $gateway } from './gateway'
import { $nativeNotifyPrefs, setNativeNotifyEnabled, setNativeNotifyKind } from './native-notifications'
import {
  createPresenceReporter,
  presenceEvidence,
  type PresenceInputs,
  type PresenceReporterDeps,
  startPresenceReporter
} from './presence'
import { $activeSessionId, $selectedStoredSessionId } from './session'
import { $sessionStates } from './session-states'

const NOW = Date.UTC(2026, 10, 20, 12)

function fakeChannel(clientActivity = true) {
  const calls: Record<string, unknown>[] = []

  return {
    calls,
    clientActivity,
    connectionState: 'open',
    request: vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method === 'client.activity') {
        calls.push(params)
      }

      return {} as never
    })
  }
}

let seq = 0

function fakeSystem(initial: Pick<PresenceSnapshot, 'idleSeconds' | 'locked' | 'suspended' | 'system'>) {
  let snapshot: PresenceSnapshot = { ...initial, sampledAt: Date.now(), seq: ++seq }
  const listeners = new Set<(next: PresenceSnapshot) => void>()

  return {
    get: vi.fn(async () => snapshot),
    onChange: (callback: (next: PresenceSnapshot) => void) => {
      listeners.add(callback)

      return () => void listeners.delete(callback)
    },
    // What main would push; `get` keeps answering the newest value.
    push: (next: Partial<PresenceSnapshot>) => {
      snapshot = { ...snapshot, sampledAt: Date.now(), seq: ++seq, ...next }
      listeners.forEach(listener => listener(snapshot))
    },
    set: (next: Partial<PresenceSnapshot>) => void (snapshot = { ...snapshot, sampledAt: Date.now(), seq: ++seq, ...next })
  }
}

interface View {
  canAlert: boolean
  focused: boolean
  sessionId: null | string
  visible: boolean
}

function harness({
  channel = fakeChannel(),
  system,
  view: initial = {}
}: { channel?: ReturnType<typeof fakeChannel>; system?: ReturnType<typeof fakeSystem>; view?: Partial<View> } = {}) {
  const view: View = { canAlert: true, focused: true, sessionId: 's1', visible: true, ...initial }
  const target = new EventTarget()

  const deps: PresenceReporterDeps = {
    canAlert: () => view.canAlert,
    focused: () => view.focused,
    inputTarget: target,
    system,
    view: () => (view.sessionId ? { channel, kind: 'session', sessionId: view.sessionId } : { channel, kind: 'none' }),
    visible: () => view.visible
  }

  const reporter = createPresenceReporter(deps)

  return {
    channel,
    reporter,
    view,
    input: () => target.dispatchEvent(new Event('keydown')),
    settle: async (ms = 0) => {
      await vi.advanceTimersByTimeAsync(ms)
    }
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW)
})

afterEach(() => void vi.useRealTimers())

const page = (overrides: Partial<PresenceInputs> = {}): PresenceInputs => ({
  focused: true,
  lastPageInputAt: NOW - 10_000,
  locked: false,
  now: NOW,
  suspended: false,
  system: false,
  systemIdleSeconds: null,
  visible: true,
  ...overrides
})

describe('presenceEvidence', () => {
  it('uses the OS idle clock whatever this window is doing, and page input otherwise', () => {
    expect(presenceEvidence(page({ system: true, systemIdleSeconds: 5, visible: false, focused: false }))).toEqual({
      active: true,
      basis: 'system',
      idleSeconds: 5
    })
    expect(presenceEvidence(page({ system: true, systemIdleSeconds: 5, locked: true })).active).toBe(false)
    expect(presenceEvidence(page({ system: true, systemIdleSeconds: 5, suspended: true })).active).toBe(false)
    expect(presenceEvidence(page({ system: true, systemIdleSeconds: 61 })).active).toBe(false)

    expect(presenceEvidence(page())).toEqual({ active: true, basis: 'page', idleSeconds: 10 })
    expect(presenceEvidence(page({ lastPageInputAt: NOW - 61_000 })).active).toBe(false)
    // Focus and visibility are never evidence: no input means nothing to report.
    expect(presenceEvidence(page({ lastPageInputAt: null }))).toEqual({ active: false, basis: 'page', idleSeconds: null })
    expect(presenceEvidence(page({ visible: false })).active).toBe(false)
  })
})

describe('createPresenceReporter', () => {
  it('reports once when the user becomes active, then at most once per 45 seconds', async () => {
    const { channel, input, settle } = harness()

    await settle()
    expect(channel.calls).toEqual([])

    input()
    await settle()
    expect(channel.calls).toEqual([{ basis: 'page', can_alert: true, idle_seconds: 0, sessions: ['s1'] }])

    for (let second = 0; second < 44; second += 1) {
      input()
      await settle(1000)
    }

    expect(channel.calls).toHaveLength(1)
    input()
    await settle(15_000)
    expect(channel.calls).toHaveLength(2)

    // Walking away: no evidence, no reports.
    await settle(300_000)
    expect(channel.calls).toHaveLength(3)
    expect(channel.calls.every(call => (call.idle_seconds as number) < 60)).toBe(true)
  })

  it('sends nothing to a backend that did not advertise client_activity', async () => {
    const { channel, input, settle } = harness({ channel: fakeChannel(false) })

    input()
    await settle(120_000)

    expect(channel.request).not.toHaveBeenCalled()
  })

  it('reports nothing without a session in view', async () => {
    const { channel, input, settle } = harness({ view: { sessionId: null } })

    input()
    await settle(60_000)

    expect(channel.calls).toEqual([])
  })

  it('page tier ignores input while the document is hidden', async () => {
    const { channel, input, settle, view } = harness({ view: { visible: false } })

    input()
    await settle(60_000)
    view.visible = true
    await settle(15_000)

    expect(channel.calls).toEqual([])
  })

  it('system tier reports while the window is unfocused and hidden', async () => {
    const system = fakeSystem({ system: true, idleSeconds: 3, locked: false, suspended: false })
    const { channel, settle } = harness({ system, view: { canAlert: false, focused: false, visible: false } })

    await settle()

    expect(channel.calls).toEqual([{ basis: 'system', can_alert: false, idle_seconds: 3, sessions: ['s1'] }])
  })

  it('system tier: input in this window after an idle spell asks the OS clock again at once', async () => {
    const system = fakeSystem({ system: true, idleSeconds: 120, locked: false, suspended: false })
    const { channel, input, settle } = harness({ system })

    await settle()
    expect(channel.calls).toEqual([])

    // The user is back and touches this window: the OS clock restarts, and main has not polled yet.
    system.set({ idleSeconds: 0 })
    input()
    await settle(100)
    // Not before the OS has had a moment to count that input.
    expect(channel.calls).toEqual([])
    await settle(200)

    // The fresh sample dates from the input, a quarter of a second ago.
    expect(channel.calls).toEqual([{ basis: 'system', can_alert: true, idle_seconds: 0.25, sessions: ['s1'] }])
  })

  it('system tier: input in this window is never evidence, and asks the OS clock at most once a second', async () => {
    // Input the OS does not count (a remote-control tool, a script): the OS clock keeps saying idle.
    const system = fakeSystem({ system: true, idleSeconds: 120, locked: false, suspended: false })
    const { channel, input, settle } = harness({ system })

    await settle()
    const asked = system.get.mock.calls.length

    for (let tenth = 0; tenth < 50; tenth += 1) {
      input()
      await settle(100)
    }

    expect(channel.calls).toEqual([])
    expect(system.get.mock.calls.length - asked).toBeGreaterThanOrEqual(4)
    expect(system.get.mock.calls.length - asked).toBeLessThanOrEqual(6)
  })

  it('system tier: input while the user is already active asks nothing extra', async () => {
    const system = fakeSystem({ system: true, idleSeconds: 3, locked: false, suspended: false })
    const { channel, input, settle } = harness({ system })

    await settle()
    const asked = system.get.mock.calls.length

    for (let tenth = 0; tenth < 50; tenth += 1) {
      input()
      await settle(100)
    }

    expect(system.get.mock.calls.length).toBe(asked)
    expect(channel.calls).toHaveLength(1)
  })

  it('sends away exactly once on lock, and again only after a later active report', async () => {
    const system = fakeSystem({ system: true, idleSeconds: 2, locked: false, suspended: false })
    const { channel, settle } = harness({ system })

    await settle()
    system.push({ locked: true })
    system.push({ suspended: true })
    await settle(120_000)

    const away = expect.objectContaining({ away: true, basis: 'system', sessions: [] })
    expect(channel.calls).toEqual([expect.objectContaining({ sessions: ['s1'] }), away])

    system.push({ idleSeconds: 1, locked: false, suspended: false })
    system.push({ suspended: true })
    expect(channel.calls).toEqual([expect.objectContaining({ sessions: ['s1'] }), away, expect.objectContaining({ sessions: ['s1'] }), away])
  })

  it('reports at once when the session in view changes or can_alert flips', async () => {
    const system = fakeSystem({ system: true, idleSeconds: 1, locked: false, suspended: false })
    const { channel, reporter, settle, view } = harness({ system })

    await settle()
    view.sessionId = 's2'
    reporter.refresh()
    await settle()
    reporter.refresh()
    await settle()
    view.canAlert = false
    reporter.refresh()
    await settle()

    expect(channel.calls.map(call => [call.sessions, call.can_alert])).toEqual([
      [['s1'], true],
      [['s2'], true],
      [['s2'], false]
    ])
  })

  it('swallows a refused report', async () => {
    const channel = fakeChannel()
    channel.request.mockRejectedValue(new Error('invalid params for client.activity'))
    const system = fakeSystem({ system: true, idleSeconds: 1, locked: false, suspended: false })
    const { settle } = harness({ channel, system })

    await settle(60_000)

    expect(channel.request).toHaveBeenCalled()
  })
})

describe('can_alert from this window', () => {
  const cases = [
    // [host, focused, turn-done notification on, browser permission, can_alert]
    ['desktop', true, false, 'default', true],
    ['desktop', false, true, 'default', true],
    ['desktop', false, false, 'default', false],
    ['webapp', true, false, 'denied', true],
    ['webapp', false, true, 'granted', true],
    ['webapp', false, true, 'denied', false],
    ['webapp', false, false, 'granted', false]
  ] as const

  afterEach(() => {
    delete document.documentElement.dataset.hermesDesktopHost
    vi.unstubAllGlobals()
    window.location.hash = ''
    $gateway.set(null)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $sessionStates.set({})
    setNativeNotifyEnabled(true)
    setNativeNotifyKind('turnDone', true)
  })

  it.each(cases)('%s, focused %s, turn-done %s, permission %s -> %s', async (host, focused, turnDone, permission, expected) => {
    if (host === 'webapp') {
      document.documentElement.dataset.hermesDesktopHost = 'browser'
      // The live permission the browser bridge checks, not only the cached store.
      vi.stubGlobal('isSecureContext', true)
      vi.stubGlobal('Notification', { permission })
    }

    vi.spyOn(document, 'hasFocus').mockReturnValue(focused)
    setNativeNotifyKind('turnDone', turnDone)
    $browserNotificationPermission.set(permission)
    expect($nativeNotifyPrefs.get().kinds.turnDone).toBe(turnDone)

    const channel = fakeChannel()
    $gateway.set(channel as never)
    window.location.hash = '#/stored-1'
    $selectedStoredSessionId.set('stored-1')
    $activeSessionId.set('s1')
    $sessionStates.set({ s1: createClientSessionState('stored-1') })
    const stop = startPresenceReporter()

    document.dispatchEvent(new Event('pointerdown'))
    await vi.advanceTimersByTimeAsync(0)
    stop()

    expect(channel.calls).toEqual([{ basis: 'page', can_alert: expected, idle_seconds: 0, sessions: ['s1'] }])
  })
})
