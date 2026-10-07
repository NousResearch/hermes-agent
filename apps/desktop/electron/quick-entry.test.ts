import { describe, expect, it, vi } from 'vitest'

import {
  createQuickEntryShortcut,
  createQuickEntrySubmitRelay,
  DEFAULT_QUICK_ENTRY_SHORTCUT,
  type GlobalShortcutLike,
  parseQuickEntryShortcut,
  quickEntryPositionFromBounds,
  quickEntryWindowBounds,
  sanitizeQuickEntrySettings
} from './quick-entry'

describe('quick-entry submit ack relay', () => {
  it('ignores an ack for a stale correlation id', async () => {
    const relay = createQuickEntrySubmitRelay({ onSuccess: vi.fn(), timeoutMs: 1000 })
    let correlationId = ''
    let settled = false

    const pending = relay
      .begin(id => {
        correlationId = id
      })
      .then(() => {
        settled = true
      })

    relay.acknowledge('qe-stale', { ok: true })
    await Promise.resolve()

    expect(settled).toBe(false)
    expect(relay.pendingCount()).toBe(1)
    relay.acknowledge(correlationId, { ok: false })
    await pending
    expect(relay.pendingCount()).toBe(0)
  })

  it('resolves an unacknowledged submit as an unknown, non-retryable timeout and keeps it reconcilable', async () => {
    vi.useFakeTimers()
    const relay = createQuickEntrySubmitRelay({ onSuccess: vi.fn(), timeoutMs: 15000 })
    const pending = relay.begin(vi.fn())

    await vi.advanceTimersByTimeAsync(15000)

    await expect(pending).resolves.toMatchObject({ code: 'timeout', ok: false, retryable: false })
    expect(relay.pendingCount()).toBe(0)
    expect(relay.reconcilableCount()).toBe(1)
    vi.useRealTimers()
  })

  it('reconciles a late success through onLateResult without hiding the window', async () => {
    vi.useFakeTimers()
    const onSuccess = vi.fn()
    const onLateResult = vi.fn()
    const relay = createQuickEntrySubmitRelay({ onLateResult, onSuccess, timeoutMs: 15000 })
    let correlationId = ''

    const pending = relay.begin(id => {
      correlationId = id
    })

    await vi.advanceTimersByTimeAsync(15000)
    await pending
    relay.acknowledge(correlationId, { ok: true })

    expect(onLateResult).toHaveBeenCalledTimes(1)
    expect(onLateResult).toHaveBeenCalledWith(correlationId, { ok: true })
    expect(onSuccess).not.toHaveBeenCalled()
    expect(relay.reconcilableCount()).toBe(0)
    vi.useRealTimers()
  })

  it('reconciles a late failure through onLateResult', async () => {
    vi.useFakeTimers()
    const onLateResult = vi.fn()
    const relay = createQuickEntrySubmitRelay({ onLateResult, onSuccess: vi.fn(), timeoutMs: 15000 })
    let correlationId = ''

    const pending = relay.begin(id => {
      correlationId = id
    })

    await vi.advanceTimersByTimeAsync(15000)
    await pending
    relay.acknowledge(correlationId, { code: 'submit-failed', ok: false })

    expect(onLateResult).toHaveBeenCalledTimes(1)
    expect(relay.reconcilableCount()).toBe(0)
    vi.useRealTimers()
  })

  it('ignores a duplicate late ack', async () => {
    vi.useFakeTimers()
    const onLateResult = vi.fn()
    const relay = createQuickEntrySubmitRelay({ onLateResult, onSuccess: vi.fn(), timeoutMs: 15000 })
    let correlationId = ''

    const pending = relay.begin(id => {
      correlationId = id
    })

    await vi.advanceTimersByTimeAsync(15000)
    await pending
    relay.acknowledge(correlationId, { ok: true })
    relay.acknowledge(correlationId, { ok: true })

    expect(onLateResult).toHaveBeenCalledTimes(1)
    expect(relay.reconcilableCount()).toBe(0)
    vi.useRealTimers()
  })

  it('runs hide and primary focus only after an ok acknowledgement', async () => {
    const events: string[] = []

    const relay = createQuickEntrySubmitRelay({
      onSuccess: () => events.push('hide-primary-show-focus'),
      timeoutMs: 1000
    })

    let correlationId = ''

    const pending = relay.begin(id => {
      correlationId = id
      events.push('forward')
    })

    relay.acknowledge(correlationId, { ok: false })
    await expect(pending).resolves.toMatchObject({ ok: false })
    expect(events).toEqual(['forward'])

    let okCorrelationId = ''

    const okPending = relay.begin(id => {
      okCorrelationId = id
      events.push('forward-ok')
    })

    relay.acknowledge(okCorrelationId, { ok: true })
    await expect(okPending).resolves.toMatchObject({ ok: true })
    expect(events).toEqual(['forward', 'forward-ok', 'hide-primary-show-focus'])
  })
})

function fakeGlobalShortcut(options: { register?: boolean; taken?: string[] } = {}) {
  const held = new Set(options.taken ?? [])

  const globalShortcut: GlobalShortcutLike = {
    isRegistered: vi.fn((accelerator: string) => held.has(accelerator)),
    register: vi.fn((accelerator: string) => {
      if (options.register === false) {
        return false
      }

      held.add(accelerator)

      return true
    }),
    unregister: vi.fn((accelerator: string) => void held.delete(accelerator))
  }

  return { globalShortcut, held }
}

describe('parseQuickEntryShortcut', () => {
  it('normalizes casing, aliases, and modifier order', () => {
    expect(parseQuickEntryShortcut('cmdorctrl+shift+space')).toEqual({
      accelerator: 'CommandOrControl+Shift+Space',
      ok: true
    })
    expect(parseQuickEntryShortcut('  Shift + CTRL + k ')).toEqual({ accelerator: 'Control+Shift+K', ok: true })
    expect(parseQuickEntryShortcut('Alt+f5')).toEqual({ accelerator: 'Alt+F5', ok: true })
    expect(parseQuickEntryShortcut('Meta+/')).toEqual({ accelerator: 'Super+/', ok: true })
  })

  it('collapses duplicate modifiers', () => {
    expect(parseQuickEntryShortcut('Ctrl+Control+Shift+J')).toEqual({ accelerator: 'Control+Shift+J', ok: true })
  })

  it('requires a modifier so a global bind cannot swallow a bare key', () => {
    expect(parseQuickEntryShortcut('K')).toEqual({ ok: false, reason: 'no-modifier' })
    expect(parseQuickEntryShortcut('Space')).toEqual({ ok: false, reason: 'no-modifier' })
  })

  it('requires exactly one non-modifier key', () => {
    expect(parseQuickEntryShortcut('Shift+Control')).toEqual({ ok: false, reason: 'no-key' })
    expect(parseQuickEntryShortcut('Shift+A+B')).toEqual({ ok: false, reason: 'invalid-key' })
    expect(parseQuickEntryShortcut('A+Shift')).toEqual({ ok: false, reason: 'invalid-modifier' })
  })

  it('rejects empty, junk, and the reserved Escape key', () => {
    expect(parseQuickEntryShortcut('')).toEqual({ ok: false, reason: 'empty' })
    expect(parseQuickEntryShortcut('   ')).toEqual({ ok: false, reason: 'empty' })
    expect(parseQuickEntryShortcut(null)).toEqual({ ok: false, reason: 'empty' })
    expect(parseQuickEntryShortcut('Ctrl+NotAKey')).toEqual({ ok: false, reason: 'invalid-key' })
    // Escape hides the window; binding it globally would make it un-toggleable.
    expect(parseQuickEntryShortcut('Ctrl+Escape')).toEqual({ ok: false, reason: 'reserved' })
  })

  it('accepts the shipped default unchanged', () => {
    expect(parseQuickEntryShortcut(DEFAULT_QUICK_ENTRY_SHORTCUT)).toEqual({
      accelerator: DEFAULT_QUICK_ENTRY_SHORTCUT,
      ok: true
    })
  })
})

describe('sanitizeQuickEntrySettings', () => {
  it('defaults to enabled with the default shortcut and the shipped position', () => {
    const shipped = { enabled: true, position: { x: 0.5, y: 0.22 }, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT }

    expect(sanitizeQuickEntrySettings(undefined)).toEqual(shipped)
    expect(sanitizeQuickEntrySettings('not an object')).toEqual(shipped)
  })

  it('keeps an explicit disable and normalizes a stored shortcut', () => {
    expect(sanitizeQuickEntrySettings({ enabled: false, shortcut: 'alt+j' })).toEqual({
      enabled: false,
      position: { x: 0.5, y: 0.22 },
      shortcut: 'Alt+J'
    })
  })

  it('keeps a saved position and falls back per axis when one is missing', () => {
    expect(sanitizeQuickEntrySettings({ position: { x: 0.8, y: 0.1 } }).position).toEqual({ x: 0.8, y: 0.1 })
    expect(sanitizeQuickEntrySettings({ position: { x: 0.8 } }).position).toEqual({ x: 0.8, y: 0.22 })
    expect(sanitizeQuickEntrySettings({ position: { y: 0.9 } }).position).toEqual({ x: 0.5, y: 0.9 })
  })

  it('clamps an out-of-range position instead of dropping it', () => {
    expect(sanitizeQuickEntrySettings({ position: { x: 1.4, y: -0.25 } }).position).toEqual({ x: 1, y: 0 })
  })

  it('falls back to the shipped position for junk values', () => {
    const shipped = { x: 0.5, y: 0.22 }

    expect(sanitizeQuickEntrySettings({ position: { x: '0.8', y: Number.NaN } }).position).toEqual(shipped)
    expect(sanitizeQuickEntrySettings({ position: { x: Number.POSITIVE_INFINITY } }).position).toEqual(shipped)
    expect(sanitizeQuickEntrySettings({ position: 'center' }).position).toEqual(shipped)
    expect(sanitizeQuickEntrySettings({ position: null }).position).toEqual(shipped)
  })

  it('round-trips a sanitized settings object unchanged (legacy file included)', () => {
    // A pre-position file ({enabled, shortcut} only) sanitizes to defaults and
    // then survives a write/read cycle byte-for-byte.
    const legacy = sanitizeQuickEntrySettings({ enabled: false, shortcut: 'Alt+J' })

    expect(legacy.position).toEqual({ x: 0.5, y: 0.22 })
    expect(sanitizeQuickEntrySettings(legacy)).toEqual(legacy)
    expect(sanitizeQuickEntrySettings({ ...legacy, position: { x: 0.7, y: 0.4 } })).toEqual({
      ...legacy,
      position: { x: 0.7, y: 0.4 }
    })
  })

  it('falls back to the default when the stored shortcut is unusable', () => {
    expect(sanitizeQuickEntrySettings({ enabled: true, shortcut: 'Q' })).toEqual({
      enabled: true,
      position: { x: 0.5, y: 0.22 },
      shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT
    })
  })

  it('treats a non-boolean enabled as off (only `true` opts in once present)', () => {
    expect(sanitizeQuickEntrySettings({ enabled: 'yes' }).enabled).toBe(false)
  })
})

describe('createQuickEntryShortcut', () => {
  it('registers the normalized accelerator when enabled', () => {
    const { globalShortcut } = fakeGlobalShortcut()
    const onTrigger = vi.fn()
    const controller = createQuickEntryShortcut(globalShortcut, onTrigger)

    const state = controller.apply({ enabled: true, shortcut: 'cmdorctrl+shift+space' })

    expect(state).toEqual({ error: null, registered: true, shortcut: 'CommandOrControl+Shift+Space' })
    expect(globalShortcut.register).toHaveBeenCalledWith('CommandOrControl+Shift+Space', onTrigger)
    expect(controller.current()).toEqual(state)
  })

  it('never registers while the setting is disabled', () => {
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    const state = controller.apply({ enabled: false, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT })

    expect(globalShortcut.register).not.toHaveBeenCalled()
    expect(state).toEqual({ error: null, registered: false, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT })
  })

  it('releases the old accelerator before registering a new one', () => {
    const { globalShortcut, held } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    controller.apply({ enabled: true, shortcut: 'Alt+J' })
    controller.apply({ enabled: true, shortcut: 'Alt+K' })

    expect(globalShortcut.unregister).toHaveBeenCalledWith('Alt+J')
    expect(held.has('Alt+J')).toBe(false)
    expect(held.has('Alt+K')).toBe(true)
  })

  it('turning the feature off releases the live accelerator', () => {
    const { globalShortcut, held } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    controller.apply({ enabled: true, shortcut: 'Alt+J' })
    const off = controller.apply({ enabled: false, shortcut: 'Alt+J' })

    expect(globalShortcut.unregister).toHaveBeenCalledWith('Alt+J')
    expect(held.size).toBe(0)
    expect(off.registered).toBe(false)
    expect(off.error).toBeNull()
  })

  it("surfaces 'taken' when another app already owns the chord", () => {
    const { globalShortcut } = fakeGlobalShortcut({ taken: ['Alt+J'] })
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    const state = controller.apply({ enabled: true, shortcut: 'alt+j' })

    expect(globalShortcut.register).not.toHaveBeenCalled()
    expect(state).toEqual({ error: 'taken', registered: false, shortcut: 'Alt+J' })
  })

  it("surfaces 'taken' when the OS refuses the registration", () => {
    const { globalShortcut } = fakeGlobalShortcut({ register: false })
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    expect(controller.apply({ enabled: true, shortcut: 'Alt+J' })).toEqual({
      error: 'taken',
      registered: false,
      shortcut: 'Alt+J'
    })
  })

  it("surfaces 'invalid' for an unusable shortcut without asking the OS", () => {
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    expect(controller.apply({ enabled: true, shortcut: 'J' })).toEqual({
      error: 'invalid',
      registered: false,
      shortcut: 'J'
    })
    expect(globalShortcut.register).not.toHaveBeenCalled()
  })

  it('survives a throwing globalShortcut', () => {
    const globalShortcut: GlobalShortcutLike = {
      isRegistered: () => false,
      register: () => {
        throw new Error('x11 grab failed')
      },
      unregister: () => {}
    }

    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    expect(controller.apply({ enabled: true, shortcut: 'Alt+J' }).error).toBe('taken')
  })

  it('dispose releases the accelerator and is idempotent', () => {
    const { globalShortcut, held } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn())

    controller.apply({ enabled: true, shortcut: 'Alt+J' })
    controller.dispose()
    controller.dispose()

    expect(globalShortcut.unregister).toHaveBeenCalledTimes(1)
    expect(held.size).toBe(0)
    expect(controller.current().registered).toBe(false)
  })
})

describe('quickEntryWindowBounds', () => {
  it('keeps the shipped defaults: centred, 22% down the work area', () => {
    const area = { height: 1000, width: 1600, x: 0, y: 0 }
    const bounds = quickEntryWindowBounds(area)

    expect(bounds.width).toBe(640)
    expect(bounds.x).toBe((1600 - 640) / 2)
    // 0.22 * 1000 — the placement the feature has always shipped.
    expect(bounds.y).toBe(220)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(1000)
    expect(bounds).toEqual(quickEntryWindowBounds(area, { x: 0.5, y: 0.22 }))
  })

  it('puts the horizontal centre and the top edge on the given fractions', () => {
    const bounds = quickEntryWindowBounds({ height: 2112, width: 3840, x: 0, y: 0 }, { x: 0.75, y: 0.5 })

    // Centre at 75% of 3840 = 2880 → left edge 2880 - 320; top at 50% of 2112.
    expect(bounds.x).toBe(2560)
    expect(bounds.y).toBe(1056)
  })

  it('clamps at all four edges of the work area', () => {
    const area = { height: 400, width: 1000, x: 100, y: 50 }

    expect(quickEntryWindowBounds(area, { x: 0, y: 0 })).toEqual({ height: 168, width: 640, x: 100, y: 50 })

    const bottomRight = quickEntryWindowBounds(area, { x: 1, y: 1 })
    expect(bottomRight.x).toBe(100 + 1000 - 640)
    expect(bottomRight.y).toBe(50 + 400 - 168)

    // Fractions outside 0..1 cannot push any part of the window off-screen.
    const outside = quickEntryWindowBounds(area, { x: -0.5, y: 1.5 })
    expect(outside.x).toBe(100)
    expect(outside.y).toBe(50 + 400 - 168)
  })

  it('respects a display origin offset (second monitor)', () => {
    const bounds = quickEntryWindowBounds({ height: 900, width: 1440, x: 1600, y: -200 })

    expect(bounds.x).toBe(1600 + (1440 - 640) / 2)
    expect(bounds.y).toBeGreaterThanOrEqual(-200)
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(1600 + 1440)
  })

  it('stays inside a tiny work area', () => {
    const bounds = quickEntryWindowBounds({ height: 120, width: 320, x: 0, y: 0 }, { x: 1, y: 1 })

    expect(bounds.width).toBeLessThanOrEqual(320)
    expect(bounds.height).toBeLessThanOrEqual(120)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(120)
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(320)
  })

  it('falls back to the origin without a work area', () => {
    expect(quickEntryWindowBounds()).toEqual({ height: 168, width: 640, x: 0, y: 0 })
    expect(quickEntryWindowBounds(undefined, { x: 1, y: 1 })).toEqual({ height: 168, width: 640, x: 0, y: 0 })
  })
})

describe('quickEntryPositionFromBounds', () => {
  type Box = { height: number; width: number; x: number; y: number }

  it('round-trips bounds placed by quickEntryWindowBounds on any display', () => {
    // The persisted shape must survive the trip back into a placement: measure
    // where the window sits, feed the fractions back, get the same pixels —
    // otherwise every re-summon would creep the window by a pixel.
    const cases: [Box, Box][] = [
      // Shipped default, and both extremes of one display.
      [
        { height: 1000, width: 1600, x: 0, y: 0 },
        { height: 168, width: 640, x: 480, y: 220 }
      ],
      [
        { height: 1000, width: 1600, x: 0, y: 0 },
        { height: 168, width: 640, x: 0, y: 0 }
      ],
      [
        { height: 1000, width: 1600, x: 0, y: 0 },
        { height: 168, width: 640, x: 960, y: 832 }
      ],
      // A 4K display with a custom placement.
      [
        { height: 2112, width: 3840, x: 0, y: 0 },
        { height: 168, width: 640, x: 2560, y: 1056 }
      ],
      // A second monitor with a non-zero origin, including a negative one.
      [
        { height: 1080, width: 1920, x: 3840, y: 0 },
        { height: 168, width: 640, x: 4000, y: 500 }
      ],
      [
        { height: 2560, width: 1440, x: -1440, y: -200 },
        { height: 168, width: 640, x: -1040, y: 1000 }
      ]
    ]

    for (const [workArea, bounds] of cases) {
      expect(quickEntryWindowBounds(workArea, quickEntryPositionFromBounds(bounds, workArea))).toEqual(bounds)
    }
  })

  it('clamps a window hanging off any edge back into 0..1', () => {
    const workArea = { height: 800, width: 1200, x: 0, y: 0 }

    expect(quickEntryPositionFromBounds({ height: 168, width: 640, x: -400, y: -120 }, workArea)).toEqual({
      x: 0,
      y: 0
    })
    expect(quickEntryPositionFromBounds({ height: 168, width: 640, x: 1000, y: 900 }, workArea)).toEqual({ x: 1, y: 1 })
  })

  it('measures against a non-zero work-area origin (second monitor)', () => {
    const workArea = { height: 1440, width: 2560, x: 3840, y: 200 }

    // Parked in the work area's top-left corner: its centre sits half a window
    // (320 of 2560) inside the origin, and its top edge exactly on it.
    expect(quickEntryPositionFromBounds({ height: 168, width: 640, x: 3840, y: 200 }, workArea)).toEqual({
      x: 0.125,
      y: 0
    })
    expect(quickEntryPositionFromBounds({ height: 168, width: 640, x: 4800, y: 344 }, workArea)).toEqual({
      x: 0.5,
      y: 0.1
    })
  })

  it('falls back to the shipped placement for a degenerate work area', () => {
    const bounds = { height: 168, width: 640, x: 0, y: 0 }

    expect(quickEntryPositionFromBounds(bounds, { height: 0, width: 0, x: 0, y: 0 })).toEqual({ x: 0.5, y: 0.22 })
  })
})
