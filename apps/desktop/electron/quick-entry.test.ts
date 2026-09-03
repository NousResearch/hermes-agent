import { describe, expect, it, vi } from 'vitest'

import {
  canShowQuickEntryFrom,
  createQuickEntryShortcut,
  DEFAULT_QUICK_ENTRY_SHORTCUT,
  type GlobalShortcutLike,
  isQuickEntryAgentOffered,
  isQuickEntryFallbackProfile,
  parseQuickEntryShortcut,
  quickEntryRejectedLaunchResult,
  quickEntryScreenAnchorRect,
  quickEntryWindowBounds,
  sanitizeQuickEntrySettings
} from './quick-entry'

describe('canShowQuickEntryFrom', () => {
  it('trusts only live declared pet hosts', () => {
    const primarySender = { id: 'primary' }
    const overlaySender = { id: 'overlay' }
    const destroyedSender = { id: 'destroyed' }

    const hosts = [
      { isDestroyed: () => false, webContents: primarySender },
      { isDestroyed: () => false, webContents: overlaySender },
      { isDestroyed: () => true, webContents: destroyedSender }
    ]

    expect(canShowQuickEntryFrom(primarySender, hosts)).toBe(true)
    expect(canShowQuickEntryFrom(overlaySender, hosts)).toBe(true)
    expect(canShowQuickEntryFrom(destroyedSender, hosts)).toBe(false)
    expect(canShowQuickEntryFrom({ id: 'quick-entry' }, hosts)).toBe(false)
    expect(canShowQuickEntryFrom(primarySender, [null, undefined])).toBe(false)
  })
})

describe('quickEntryScreenAnchorRect', () => {
  it('translates a renderer-local pet rectangle into screen coordinates', () => {
    expect(
      quickEntryScreenAnchorRect(
        { x: 300, y: 200 },
        { height: 80, width: 56, x: 24, y: 40 }
      )
    ).toEqual({ height: 80, width: 56, x: 324, y: 240 })
  })

  it('scales renderer coordinates into Electron content bounds at non-default zoom', () => {
    expect(
      quickEntryScreenAnchorRect(
        { height: 800, width: 1200, x: 100, y: 50 },
        { height: 80, viewportHeight: 1000, viewportWidth: 1600, width: 56, x: 200, y: 700 }
      )
    ).toEqual({ height: 64, width: 42, x: 250, y: 610 })
  })

  it('rejects an invalid rectangle', () => {
    expect(quickEntryScreenAnchorRect({ x: 0, y: 0 }, { height: 0, width: 56, x: 24, y: 40 })).toBeUndefined()
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
  it('defaults to enabled with the default shortcut', () => {
    expect(sanitizeQuickEntrySettings(undefined)).toEqual({ enabled: true, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT })
    expect(sanitizeQuickEntrySettings('not an object')).toEqual({
      enabled: true,
      shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT
    })
  })

  it('keeps an explicit disable and normalizes a stored shortcut', () => {
    expect(sanitizeQuickEntrySettings({ enabled: false, shortcut: 'alt+j' })).toEqual({
      enabled: false,
      shortcut: 'Alt+J'
    })
  })

  it('falls back to the default when the stored shortcut is unusable', () => {
    expect(sanitizeQuickEntrySettings({ enabled: true, shortcut: 'Q' })).toEqual({
      enabled: true,
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
  it('centers horizontally and sits below the top edge of the work area', () => {
    const bounds = quickEntryWindowBounds({ height: 1000, width: 1600, x: 0, y: 0 })

    expect(bounds.width).toBe(760)
    expect(bounds.x).toBe((1600 - 760) / 2)
    expect(bounds.y).toBeGreaterThan(0)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(1000)
  })

  it('respects a display origin offset (second monitor)', () => {
    const bounds = quickEntryWindowBounds({ height: 900, width: 1440, x: 1600, y: -200 })

    expect(bounds.x).toBe(1600 + (1440 - 760) / 2)
    expect(bounds.y).toBeGreaterThanOrEqual(-200)
  })

  it('opens beside the pointer when there is room', () => {
    const bounds = quickEntryWindowBounds({ height: 1000, width: 1600, x: 0, y: 0 }, { x: 400, y: 300 })

    expect(bounds.x).toBe(418)
    expect(bounds.y).toBe(318)
  })

  it('flips around the pointer and clamps on monitor edges', () => {
    const bounds = quickEntryWindowBounds({ height: 900, width: 1440, x: -1440, y: -200 }, { x: -20, y: 650 })

    expect(bounds.x).toBe(-798)
    expect(bounds.y).toBe(212)
    expect(bounds.x).toBeGreaterThanOrEqual(-1440)
    expect(bounds.y).toBeGreaterThanOrEqual(-200)
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(0)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(700)
  })

  it('stays inside a tiny work area', () => {
    const bounds = quickEntryWindowBounds({ height: 120, width: 320, x: 0, y: 0 })

    expect(bounds.width).toBeLessThanOrEqual(320)
    expect(bounds.height).toBeLessThanOrEqual(120)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(120)
  })

  it('falls back to the origin without a work area', () => {
    expect(quickEntryWindowBounds()).toEqual({ height: 420, width: 760, x: 0, y: 0 })
  })

  it('uses the compact rounded-picker footprint for a pet agent launch', () => {
    expect(quickEntryWindowBounds(undefined, undefined, 'agents')).toEqual({ height: 238, width: 224, x: 0, y: 0 })

    // Centred on the summon point and sitting above it — 500 - 224/2 = 388,
    // 400 - 238 - 10 = 152 — so the menu reads as attached to the pet.
    const bounds = quickEntryWindowBounds({ height: 900, width: 1440, x: 0, y: 0 }, { x: 500, y: 400 }, 'agents')

    expect(bounds).toEqual({ height: 238, width: 224, x: 388, y: 152 })
  })

  it('keeps the pet chooser above the summon point and falls below only at the top edge', () => {
    // Room above: opens above, and clamps X at the left edge rather than
    // letting the centred panel run off-screen.
    expect(quickEntryWindowBounds({ height: 900, width: 1440, x: 0, y: 0 }, { x: 80, y: 700 }, 'agents')).toEqual({
      height: 238,
      width: 224,
      x: 0,
      y: 452
    })

    // No room above: drops below the summon point.
    expect(quickEntryWindowBounds({ height: 900, width: 1440, x: 0, y: 0 }, { x: 500, y: 40 }, 'agents')).toEqual({
      height: 238,
      width: 224,
      x: 388,
      y: 50
    })
  })

  it('never covers the summon point while either side has room', () => {
    const workArea = { height: 900, width: 1440, x: 0, y: 0 }

    for (const y of [30, 120, 200, 450, 700, 860]) {
      const bounds = quickEntryWindowBounds(workArea, { x: 500, y }, 'agents')
      const clearsAbove = bounds.y + bounds.height <= y
      const clearsBelow = bounds.y >= y

      expect(clearsAbove || clearsBelow).toBe(true)
      expect(bounds.y).toBeGreaterThanOrEqual(workArea.y)
      expect(bounds.y + bounds.height).toBeLessThanOrEqual(workArea.y + workArea.height)
    }
  })

  it('takes the roomier side when the panel fits on neither', () => {
    // A 200px work area cannot hold the full vertical list clear of a pointer
    // in the middle, so it must pick the roomier side and shrink to fit.
    const workArea = { height: 200, width: 1440, x: 0, y: 0 }

    // Pointer low: more room above, so the panel is pushed as high as it goes.
    expect(quickEntryWindowBounds(workArea, { x: 500, y: 150 }, 'agents').y).toBe(0)
    // Pointer high: more room below, so it is pushed as low as it goes.
    expect(quickEntryWindowBounds(workArea, { x: 500, y: 40 }, 'agents').y).toBe(0)
  })

  it('respects a second monitor origin for the pet chooser', () => {
    const bounds = quickEntryWindowBounds(
      { height: 900, width: 1440, x: -1440, y: -200 },
      { x: -700, y: 400 },
      'agents'
    )

    expect(bounds).toEqual({ height: 238, width: 224, x: -812, y: 152 })
    expect(bounds.x).toBeGreaterThanOrEqual(-1440)
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(0)
  })

  it('anchors the pet chooser to the whole pet rectangle instead of the cursor point', () => {
    const bounds = quickEntryWindowBounds(
      { height: 900, width: 1440, x: 0, y: 0 },
      { x: 438, y: 340 },
      'agents',
      { height: 80, width: 56, x: 410, y: 300 }
    )

    expect(bounds).toEqual({ height: 238, width: 224, x: 326, y: 52 })
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(300)
  })

  it('shrinks on a short work area rather than overlapping the pet rectangle', () => {
    const anchor = { height: 60, width: 56, x: 132, y: 80 }

    const bounds = quickEntryWindowBounds(
      { height: 220, width: 320, x: 0, y: 0 },
      { x: 160, y: 110 },
      'agents',
      anchor
    )

    const clearsAbove = bounds.y + bounds.height <= anchor.y
    const clearsBelow = bounds.y >= anchor.y + anchor.height

    expect(clearsAbove || clearsBelow).toBe(true)
    expect(bounds.height).toBeLessThan(238)
  })
})

describe('isQuickEntryFallbackProfile', () => {
  it('allows only the shipped pet agents while the live roster is loading', () => {
    expect(isQuickEntryFallbackProfile('default')).toBe(true)
    expect(isQuickEntryFallbackProfile(' DEFAULT ')).toBe(true)
    expect(isQuickEntryFallbackProfile('jarvis')).toBe(false)
    expect(isQuickEntryFallbackProfile('invented-agent')).toBe(false)
  })
})

describe('isQuickEntryAgentOffered', () => {
  it('uses the shipped fallback only before any roster arrives', () => {
    expect(isQuickEntryAgentOffered('default', undefined)).toBe(true)
    expect(isQuickEntryAgentOffered('jarvis', undefined)).toBe(false)
  })

  it('treats an arrived roster as authoritative, including unreachable fallback agents', () => {
    expect(isQuickEntryAgentOffered('jarvis', [{ profile: 'jarvis', reachable: false }])).toBe(false)
    expect(isQuickEntryAgentOffered('jarvis', [])).toBe(false)
    expect(isQuickEntryAgentOffered('jarvis', [{ profile: 'jarvis', reachable: true }])).toBe(true)
  })
})

describe('quickEntryRejectedLaunchResult', () => {
  it('creates a correlated visible error for a rejected agent launch', () => {
    expect(
      quickEntryRejectedLaunchResult(
        { action: 'open-agent', profile: 'jarvis', requestId: 'request-1234' },
        'This launcher is no longer active.'
      )
    ).toEqual({
      error: 'This launcher is no longer active.',
      ok: false,
      profile: 'jarvis',
      requestId: 'request-1234'
    })
  })

  it('does not invent a launch result for a plain prompt submit', () => {
    expect(quickEntryRejectedLaunchResult({ text: 'hello' }, 'Rejected')).toBeUndefined()
  })
})
