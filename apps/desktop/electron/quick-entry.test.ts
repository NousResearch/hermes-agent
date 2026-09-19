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
  it('registers the normalized accelerator when enabled', async () => {
    const { globalShortcut } = fakeGlobalShortcut()
    const onTrigger = vi.fn()
    const controller = createQuickEntryShortcut(globalShortcut, onTrigger, async () => ({ available: true }))

    const state = await controller.apply({ enabled: true, shortcut: 'cmdorctrl+shift+space' })

    expect(state).toEqual({ error: null, registered: true, shortcut: 'CommandOrControl+Shift+Space' })
    expect(globalShortcut.register).toHaveBeenCalledWith('CommandOrControl+Shift+Space', onTrigger)
    expect(controller.current()).toEqual(state)
  })

  it('never registers while the setting is disabled', async () => {
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: false }))

    const state = await controller.apply({ enabled: false, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT })

    expect(globalShortcut.register).not.toHaveBeenCalled()
    expect(state).toEqual({ error: null, registered: false, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT })
  })

  it('releases the old accelerator before registering a new one', async () => {
    const { globalShortcut, held } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: true }))

    await controller.apply({ enabled: true, shortcut: 'Alt+J' })
    await controller.apply({ enabled: true, shortcut: 'Alt+K' })

    expect(globalShortcut.unregister).toHaveBeenCalledWith('Alt+J')
    expect(held.has('Alt+J')).toBe(false)
    expect(held.has('Alt+K')).toBe(true)
  })

  it('turning the feature off releases the live accelerator', async () => {
    const { globalShortcut, held } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: true }))

    controller.apply({ enabled: true, shortcut: 'Alt+J' })
    const off = await controller.apply({ enabled: false, shortcut: 'Alt+J' })

    expect(globalShortcut.unregister).toHaveBeenCalledWith('Alt+J')
    expect(held.size).toBe(0)
    expect(off.registered).toBe(false)
    expect(off.error).toBeNull()
  })

  it("surfaces 'taken' when another app already owns the chord", async () => {
    const { globalShortcut } = fakeGlobalShortcut({ taken: ['Alt+J'] })
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: false }))

    const state = await controller.apply({ enabled: true, shortcut: 'alt+j' })

    expect(globalShortcut.register).not.toHaveBeenCalled()
    expect(state).toEqual({ error: 'taken', registered: false, shortcut: 'Alt+J' })
  })

  it("surfaces 'taken' when the OS refuses the registration", async () => {
    const { globalShortcut } = fakeGlobalShortcut({ register: false })
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: true }))

    expect(await controller.apply({ enabled: true, shortcut: 'Alt+J' })).toEqual({
      error: 'taken',
      registered: false,
      shortcut: 'Alt+J'
    })
  })

  it("surfaces 'invalid' for an unusable shortcut without asking the OS", async () => {
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: false }))

    expect(await controller.apply({ enabled: true, shortcut: 'J' })).toEqual({
      error: 'invalid',
      registered: false,
      shortcut: 'J'
    })
    expect(globalShortcut.register).not.toHaveBeenCalled()
  })

  it('survives a throwing globalShortcut', async () => {
    const globalShortcut: GlobalShortcutLike = {
      isRegistered: () => false,
      register: () => {
        throw new Error('x11 grab failed')
      },
      unregister: () => {}
    }

    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: true }))

    expect((await controller.apply({ enabled: true, shortcut: 'Alt+J' })).error).toBe('taken')
  })

  it('dispose releases the accelerator and is idempotent', async () => {
    const { globalShortcut, held } = fakeGlobalShortcut()
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: true }))

    await controller.apply({ enabled: true, shortcut: 'Alt+J' })
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

// ── #95132 scenario A: a session that cannot host global shortcuts must not
// be reported as "taken by another application" ──────────────────────────────

describe('createQuickEntryShortcut portal probing (#95132)', () => {
  it("reports 'unavailable' — not 'taken' — when a Linux/Wayland registration fails and no GlobalShortcuts portal answers", async () => {
    const { globalShortcut } = fakeGlobalShortcut({ register: false })
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: false }))

    const state = await controller.apply({ enabled: true, shortcut: 'Control+Shift+Space' })

    expect(state.registered).toBe(false)
    expect(state.error).toBe('unavailable')
  })

  it("keeps reporting 'taken' when the portal probe proves the service IS reachable", async () => {
    const { globalShortcut } = fakeGlobalShortcut({ taken: ['Alt+J'] })
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({ available: true }))

    const state = await controller.apply({ enabled: true, shortcut: 'Alt+J' })

    expect(state.error).toBe('taken')
  })

  it('carries the probe detail on the unavailable state for logs and Settings', async () => {
    const { globalShortcut } = fakeGlobalShortcut({ register: false })

    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => ({
      available: false,
      detail: 'busctl: No such file or directory'
    }))

    const state = await controller.apply({ enabled: true, shortcut: 'Super+-' })

    expect(state.error).toBe('unavailable')
    expect(state.detail).toBe('busctl: No such file or directory')
  })

  it('falls back to taken when the probe itself throws', async () => {
    const { globalShortcut } = fakeGlobalShortcut({ register: false })

    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () => {
      throw new Error('probe exploded')
    })

    const state = await controller.apply({ enabled: true, shortcut: 'Alt+J' })

    expect(state.error).toBe('taken')
  })

  it('never probes for a plain conflict detected before registering', async () => {
    const { globalShortcut } = fakeGlobalShortcut({ taken: ['Alt+J'] })
    const probePortal = vi.fn(async () => ({ available: false }))
    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), probePortal)

    await controller.apply({ enabled: true, shortcut: 'Alt+J' })

    expect(probePortal).not.toHaveBeenCalled()
  })

  it('does not probe or reclassify on macOS-like platforms', async () => {
    vi.stubGlobal('process', { platform: 'darwin' })

    try {
      const { globalShortcut } = fakeGlobalShortcut({ register: false })
      const probePortal = vi.fn(async () => ({ available: false }))
      const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), probePortal)

      const state = await controller.apply({ enabled: true, shortcut: 'Alt+J' })

      expect(probePortal).not.toHaveBeenCalled()
      expect(state.error).toBe('taken')
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('releases nothing extra after an unavailable classification and can retry later', async () => {
    // The OS refuses while the session's shortcut service is down…
    let osRefuses = true
    const { globalShortcut, held } = fakeGlobalShortcut()

    globalShortcut.register = vi.fn((accelerator: string) => {
      if (osRefuses) {
        return false
      }

      held.add(accelerator)

      return true
    })

    // …and the portal comes back up afterwards (user fixed their desktop
    // session): a settings re-apply retries the registration instead of
    // staying stuck at 'unavailable'.
    let portalUp = false

    const controller = createQuickEntryShortcut(globalShortcut, vi.fn(), async () =>
      portalUp ? { available: true } : { available: false }
    )

    const first = await controller.apply({ enabled: true, shortcut: 'Alt+J' })
    expect(first.error).toBe('unavailable')

    osRefuses = false
    portalUp = true
    const second = await controller.apply({ enabled: true, shortcut: 'Alt+J' })

    expect(second.error).toBeNull()
    expect(second.registered).toBe(true)
    expect(controller.current()).toEqual(second)
  })
})
