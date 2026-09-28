import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $workspaceMode } from '@/components/pane-shell/workspace-scope'

import {
  bindDesktopMetrics,
  closeOnboardingStep,
  completeFlow,
  configPatchKeys,
  DESKTOP_METRICS_STATE_KEY,
  noteAreaClosed,
  noteBackendDrop,
  noteBackendExited,
  noteInteraction,
  noteMessageSent,
  noticeIdForToast,
  recordAction,
  recordDislike,
  recordFeatureToggle,
  recordFeatureUse,
  recordFriction,
  recordOnboarding,
  recordSettingsSaved,
  resetDesktopMetricsForTests,
  routeArea,
  setDesktopBotCount,
  setDesktopMetricsGate,
  settingsArea,
  slowFrameBucket,
  tickDesktopMetrics,
  trackArea,
  trackFlow
} from './desktop-metrics'

type Call = [string, Record<string, unknown>]

function requester(result: Record<string, unknown> = { ok: true }) {
  const calls: Call[] = []

  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    calls.push([method, params ?? {}])

    return (method === 'shared_metrics.desktop_daily' ? { recorded: true, ...result } : result) as never
  })

  return { calls, request }
}

// One record per (connection, profile); the tests' single profile owns the only one.
const stored = () => {
  const key = Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '').find(k =>
    k.startsWith(`${DESKTOP_METRICS_STATE_KEY}:`)
  )

  return key ? window.localStorage.getItem(key) : null
}

const methods = (calls: Call[]) => calls.map(([method]) => method)
const flush = () => new Promise(resolve => setTimeout(resolve, 0))

const DAY1 = Date.UTC(2026, 8, 27, 12, 0, 0)
const DAY2 = Date.UTC(2026, 8, 28, 12, 0, 0)

let bridge: { ackRendererCrashes: ReturnType<typeof vi.fn>; setEnabled: ReturnType<typeof vi.fn>; takeRendererCrashes: ReturnType<typeof vi.fn> }

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(DAY1)
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $workspaceMode.set('sessions')
  bridge = {
    ackRendererCrashes: vi.fn(async () => undefined),
    setEnabled: vi.fn(async () => undefined),
    takeRendererCrashes: vi.fn(async () => null)
  }
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { desktopMetrics: bridge }
})

afterEach(() => {
  resetDesktopMetricsForTests()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

function exerciseEverything() {
  recordFeatureUse('terminal_pane')
  trackArea('file_pane', true)
  trackArea('file_pane', false)
  recordFriction('error_toast', 'timeout')
  recordFriction('slow_frame', '250ms_to_1s')
  recordOnboarding('provider_setup', 'reached')
  recordAction('composer.send', 'click')
  recordAction('composer.send', 'click')
  recordAction('composer.send', 'click')
  recordDislike('feature_disabled', 'tips')
  recordSettingsSaved({ display: { show_reasoning: false } })
  recordFeatureToggle('tips', true, false)
  trackFlow('command_palette', true)
  trackFlow('command_palette', false)
  noteInteraction(DAY1)
  noteInteraction(DAY1 + 60_000)
  noteMessageSent('bots')
  setDesktopBotCount(3)
  noteBackendDrop(null)
}

describe('consent gate', () => {
  it('collection off: nothing is persisted and nothing is sent', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('off')
    bindDesktopMetrics(request)
    exerciseEverything()
    vi.setSystemTime(DAY2)
    tickDesktopMetrics()
    await flush()

    expect(calls).toEqual([])
    expect(stored()).toBeNull()
    expect(bridge.setEnabled).toHaveBeenCalledWith(false, expect.any(String))
    expect(bridge.takeRendererCrashes).not.toHaveBeenCalled()
  })

  it('an unknown gate (before the switch is read) records nothing either', async () => {
    const { calls, request } = requester()

    bindDesktopMetrics(request)
    exerciseEverything()
    await flush()

    expect(calls).toEqual([])
    expect(stored()).toBeNull()
  })

  it('turning collection off purges the local record and drops anything queued', async () => {
    setDesktopMetricsGate('on')
    recordFeatureUse('projects') // queued: no gateway bound yet
    recordAction('session.new', 'shortcut')
    expect(stored()).not.toBeNull()

    setDesktopMetricsGate('off')
    expect(stored()).toBeNull()

    const { calls, request } = requester()

    bindDesktopMetrics(request)
    setDesktopMetricsGate('on')
    await flush()

    expect(calls).toEqual([])
  })
})

describe('feature use', () => {
  it('counts each area once per UTC day', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)
    recordFeatureUse('terminal_pane')
    recordFeatureUse('terminal_pane')
    recordFeatureUse('settings_config_model')
    vi.setSystemTime(DAY2)
    recordFeatureUse('terminal_pane')
    await flush()

    expect(calls.filter(([m]) => m === 'shared_metrics.desktop_feature_use').map(([, p]) => p.area)).toEqual([
      'terminal_pane',
      'settings_config_model',
      'terminal_pane'
    ])
  })

  it('maps settings views and routes onto closed area ids', () => {
    expect(settingsArea('config:model')).toBe('settings_config_model')
    expect(settingsArea('connections')).toBe('settings_gateway')
    expect(settingsArea('some-plugin:page')).toBe('settings_other')
    expect(routeArea('/capabilities')).toBe('capabilities')
    expect(routeArea('/42f1c0de')).toBeNull()
    expect(routeArea('/settings')).toBeNull()
    expect(routeArea('/my-plugin', ['/my-plugin'])).toBe('extension_page')
  })
})

describe('friction', () => {
  it('buckets slow frames and caps them per day', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)

    expect(slowFrameBucket(80)).toBeNull()
    expect(slowFrameBucket(120)).toBe('100ms_to_250ms')
    expect(slowFrameBucket(6000)).toBe('gte_5s')

    for (let i = 0; i < 50; i++) {
      recordFriction('slow_frame', '1s_to_5s')
    }

    await flush()
    expect(calls.length).toBe(5)
  })

  it('maps notification ids onto closed notice ids, never the raw id', () => {
    expect(noticeIdForToast('desktop-update-available')).toBe('update_available')
    expect(noticeIdForToast('billing-block:abc123')).toBe('billing_block')
    expect(noticeIdForToast('gateway-error:/Users/me/secret')).toBe('gateway_error')
    expect(noticeIdForToast('toast-42')).toBe('other')
  })

  it('classifies a backend drop after the settle window, and a switch cancels it', async () => {
    vi.useFakeTimers({ now: DAY1 })
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)

    noteBackendDrop(null)
    noteBackendExited()
    await vi.advanceTimersByTimeAsync(3500)
    noteBackendDrop('timeout')
    await vi.advanceTimersByTimeAsync(3500)

    expect(calls.filter(([m]) => m === 'shared_metrics.desktop_friction').map(([, p]) => p.detail)).toEqual([
      'backend_exit',
      'timeout'
    ])
  })

  it('reports renderer crashes main persisted, then acks the claim', async () => {
    bridge.takeRendererCrashes.mockResolvedValueOnce({ reasons: ['oom', 'crash'] })
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)
    await flush()
    await flush()

    expect(bridge.setEnabled).toHaveBeenCalledWith(true, expect.any(String))
    expect(calls.map(([, p]) => p.detail)).toEqual(['oom', 'crash'])
    expect(bridge.ackRendererCrashes).toHaveBeenCalledWith(true)
  })
})

describe('onboarding', () => {
  it('sends each step event once per install and calls a step left open by an earlier launch abandoned', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)
    recordOnboarding('provider_setup', 'reached')
    recordOnboarding('provider_setup', 'reached')
    recordOnboarding('guide', 'reached')
    closeOnboardingStep('guide')
    await flush()
    expect(calls.map(([, p]) => `${p.step}:${p.event}`)).toEqual(['provider_setup:reached', 'guide:reached'])

    // Next launch: a fresh module (new launch id) reading the same storage.
    vi.resetModules()
    const next = await import('./desktop-metrics')
    const second = requester()

    next.setDesktopMetricsGate('on')
    next.bindDesktopMetrics(second.request)
    await flush()

    expect(second.calls.map(([, p]) => `${p.step}:${p.event}`)).toEqual(['provider_setup:abandoned'])
    next.resetDesktopMetricsForTests()
  })
})

describe('daily report: action use + mode use', () => {
  it('aggregates presses and mode time locally and sends one report per finished day', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)

    recordAction('composer.send', 'click', DAY1)
    recordAction('composer.send', 'click', DAY1 + 5000)
    recordAction('session.new', 'shortcut')
    recordAction('session.new', 'palette')
    noteInteraction(DAY1)
    noteInteraction(DAY1 + 120_000) // 2 min active
    noteInteraction(DAY1 + 120_000 + 3_600_000) // an hour idle: not active
    $workspaceMode.set('bots')
    noteInteraction(DAY1 + 120_000 + 3_600_000 + 30_000)
    noteMessageSent()
    noteMessageSent('sessions')
    setDesktopBotCount(4)
    await flush()

    expect(methods(calls)).not.toContain('shared_metrics.desktop_daily')

    vi.setSystemTime(DAY2)
    tickDesktopMetrics()
    await flush()

    const daily = calls.filter(([m]) => m === 'shared_metrics.desktop_daily')

    expect(daily).toHaveLength(1)
    const report = daily[0]![1]

    expect(report.day).toBe('2026-09-27')
    expect(report.bot_count).toBe(4)
    expect(report.actions).toEqual(
      expect.arrayContaining([
        { action: 'composer.send', count: 2, via: 'click' },
        { action: 'session.new', count: 1, via: 'shortcut' },
        { action: 'session.new', count: 1, via: 'palette' }
      ])
    )
    expect(report.modes).toEqual(
      expect.arrayContaining([
        { active_ms: 120_000, messages_sent: 1, mode: 'sessions' },
        { active_ms: 30_000, messages_sent: 1, mode: 'bots' }
      ])
    )

    // Settled → dropped: another tick sends nothing.
    tickDesktopMetrics()
    await flush()
    expect(calls.filter(([m]) => m === 'shared_metrics.desktop_daily')).toHaveLength(1)
  })

  it('keeps a finished day the backend did not record, for a retry', async () => {
    const { calls, request } = requester({ recorded: false })

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)
    recordAction('composer.send', 'shortcut')
    vi.setSystemTime(DAY2)
    tickDesktopMetrics()
    await flush()
    tickDesktopMetrics()
    await flush()

    expect(calls.filter(([m]) => m === 'shared_metrics.desktop_daily')).toHaveLength(2)
    expect(JSON.parse(stored()!).pending).toHaveLength(1)
  })
})

describe('dislike signals', () => {
  it('rage click: three clicks on one action inside a second, once per burst', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)

    for (const at of [0, 200, 400, 600, 800]) {
      recordAction('composer.send', 'click', DAY1 + at)
    }

    recordAction('composer.send', 'shortcut', DAY1 + 900)
    await flush()

    const dislikes = calls.filter(([m]) => m === 'shared_metrics.desktop_dislike').map(([, p]) => p)

    expect(dislikes).toEqual([{ signal: 'rage_click', target: 'composer.send' }])
  })

  it('quick close: an area closed within 5s of opening', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)
    trackArea('terminal_pane', true, DAY1)
    trackArea('terminal_pane', false, DAY1 + 2000)
    recordFeatureUse('review_pane', DAY1)
    noteAreaClosed('review_pane', DAY1 + 60_000)
    await flush()

    expect(calls.filter(([m]) => m === 'shared_metrics.desktop_dislike').map(([, p]) => p)).toEqual([
      { signal: 'quick_close', target: 'terminal_pane' }
    ])
  })

  it('cancelled: a flow closed without completing; a completed one is not', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)
    trackFlow('model_picker', true)
    completeFlow('model_picker')
    trackFlow('model_picker', false)
    trackFlow('command_palette', true)
    trackFlow('command_palette', false)
    await flush()

    expect(calls.map(([, p]) => p)).toEqual([{ signal: 'cancelled', target: 'command_palette' }])
  })

  it('settings: sends only the config keys, never the values; feature toggles only on the off edge', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)
    recordSettingsSaved({ display: { show_reasoning: false, skin: 'secret-skin-name' } }, 'work')
    recordFeatureToggle('tips', false, true)
    recordFeatureToggle('tips', true, false)
    await flush()

    expect(calls.map(([, p]) => p)).toEqual([
      { profile: 'work', setting: 'display.show_reasoning', signal: 'setting_off_default', target: 'setting' },
      { profile: 'work', setting: 'display.skin', signal: 'setting_off_default', target: 'setting' },
      { signal: 'feature_disabled', target: 'tips' }
    ])
    expect(JSON.stringify(calls)).not.toContain('secret-skin-name')
    expect(configPatchKeys({ a: { b: [1, 2], c: { d: null } } })).toEqual(['a.b', 'a.c.d'])
  })

  it('caps each signal per day', async () => {
    const { calls, request } = requester()

    setDesktopMetricsGate('on')
    bindDesktopMetrics(request)

    for (let i = 0; i < 100; i++) {
      recordDislike('undo', 'closed_tab')
    }

    await flush()
    expect(calls).toHaveLength(20)
  })
})

describe('per profile, per window', () => {
  it('a profile switch forgets the gate: nothing of A is kept or sent under B, and A keeps its own day', async () => {
    const a = requester()
    const b = requester()

    bindDesktopMetrics(a.request, 'local|alpha')
    setDesktopMetricsGate('on')
    recordFeatureUse('terminal_pane')
    recordAction('view.toggleSidebar', 'shortcut')

    bindDesktopMetrics(b.request, 'local|beta') // B's switch not read yet
    vi.setSystemTime(DAY2)
    tickDesktopMetrics()
    recordFeatureUse('projects')
    await flush()
    expect(b.calls).toEqual([])

    setDesktopMetricsGate('on')
    recordFeatureUse('terminal_pane')
    await flush()
    expect(b.calls).toEqual([['shared_metrics.desktop_feature_use', { area: 'terminal_pane' }]])

    bindDesktopMetrics(a.request, 'local|alpha')
    setDesktopMetricsGate('on')
    await flush()
    expect(a.calls.filter(([m]) => m === 'shared_metrics.desktop_daily').map(([, p]) => p.actions)).toEqual([
      [{ action: 'view.toggleSidebar', count: 1, via: 'shortcut' }]
    ])
  })

  it('two windows of one profile add to one day instead of overwriting each other', async () => {
    const w1 = await import('./desktop-metrics')

    w1.bindDesktopMetrics(null, 'local|default')
    w1.setDesktopMetricsGate('on')
    vi.resetModules()
    const w2 = await import('./desktop-metrics')

    w2.bindDesktopMetrics(null, 'local|default')
    w2.setDesktopMetricsGate('on')

    w1.recordAction('composer.send', 'shortcut')
    w2.recordAction('composer.send', 'shortcut')
    w2.recordAction('session.new', 'shortcut')
    w1.recordAction('composer.send', 'shortcut')

    expect(JSON.parse(stored()!).today.actions).toEqual({ 'composer.send|shortcut': 3, 'session.new|shortcut': 1 })
    w2.resetDesktopMetricsForTests()
  })
})
