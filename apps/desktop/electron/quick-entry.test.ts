import { describe, expect, it, vi } from 'vitest'

import {
  createQuickEntryShortcut,
  DEFAULT_QUICK_ENTRY_SHORTCUT,
  type GlobalShortcutLike,
  parseQuickEntryShortcut,
  quickEntryWindowBounds,
  sanitizeQuickEntrySettings
} from './quick-entry'

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

    expect(bounds.width).toBe(640)
    expect(bounds.x).toBe((1600 - 640) / 2)
    expect(bounds.y).toBeGreaterThan(0)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(1000)
  })

  it('respects a display origin offset (second monitor)', () => {
    const bounds = quickEntryWindowBounds({ height: 900, width: 1440, x: 1600, y: -200 })

    expect(bounds.x).toBe(1600 + (1440 - 640) / 2)
    expect(bounds.y).toBeGreaterThanOrEqual(-200)
  })

  it('stays inside a tiny work area', () => {
    const bounds = quickEntryWindowBounds({ height: 120, width: 320, x: 0, y: 0 })

    expect(bounds.width).toBeLessThanOrEqual(320)
    expect(bounds.height).toBeLessThanOrEqual(120)
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(120)
  })

  it('falls back to the origin without a work area', () => {
    expect(quickEntryWindowBounds()).toEqual({ height: 168, width: 640, x: 0, y: 0 })
  })
})
