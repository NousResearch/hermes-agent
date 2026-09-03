import { describe, expect, it, vi } from 'vitest'

import { createHudSummonShortcut, DEFAULT_HUD_SUMMON_SHORTCUT, resolveHudSummon } from './hud-summon-shortcut'
import type { GlobalShortcutLike } from './quick-entry'

function fakeGlobalShortcut(options: { register?: boolean; taken?: string[] } = {}) {
  const held = new Set(options.taken ?? [])
  const callbacks = new Map<string, () => void>()

  const globalShortcut: GlobalShortcutLike = {
    isRegistered: vi.fn((accelerator: string) => held.has(accelerator)),
    register: vi.fn((accelerator: string, callback: () => void) => {
      if (options.register === false || held.has(accelerator)) {
        return false
      }

      held.add(accelerator)
      callbacks.set(accelerator, callback)

      return true
    }),
    unregister: vi.fn((accelerator: string) => {
      held.delete(accelerator)
      callbacks.delete(accelerator)
    })
  }

  return { globalShortcut, held, press: (accelerator: string) => callbacks.get(accelerator)?.() }
}

describe('resolveHudSummon', () => {
  it('dismisses when the HUD is already up — the chord is a toggle', () => {
    expect(resolveHudSummon({ hudOpen: true, hermesFocused: true })).toBe('close')
    expect(resolveHudSummon({ hudOpen: true, hermesFocused: false })).toBe('close')
  })

  it('opens like the titlebar toggle when a Hermes window has focus', () => {
    expect(resolveHudSummon({ hudOpen: false, hermesFocused: true })).toBe('open-in-app')
  })

  it('opens as a companion — no focus steal, no hiding the app — when summoned from another app', () => {
    expect(resolveHudSummon({ hudOpen: false, hermesFocused: false })).toBe('open-external')
  })
})

describe('createHudSummonShortcut', () => {
  it('registers CommandOrControl+Shift+H for the life of the app', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)

    expect(controller.register()).toBe(true)
    expect(globalShortcut.isRegistered(DEFAULT_HUD_SUMMON_SHORTCUT)).toBe(true)
    expect(controller.current()).toBe(DEFAULT_HUD_SUMMON_SHORTCUT)
  })

  it('invokes the summon callback when the chord fires', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut, press } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)

    controller.register()
    press(DEFAULT_HUD_SUMMON_SHORTCUT)

    expect(summon).toHaveBeenCalledTimes(1)
  })

  it('reports taken and leaves the in-app keybind to do the job', () => {
    // Another app owns ⌘⇧H. Registration must fail loudly (false), not
    // silently claim a chord it does not hold.
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut({ taken: [DEFAULT_HUD_SUMMON_SHORTCUT] })
    const controller = createHudSummonShortcut(globalShortcut, summon)

    expect(controller.register()).toBe(false)
    expect(controller.current()).toBeNull()
  })

  it('survives an OS that refuses the registration', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut({ register: false })
    const controller = createHudSummonShortcut(globalShortcut, summon)

    expect(controller.register()).toBe(false)
    expect(controller.current()).toBeNull()
  })

  it('dispose releases the accelerator and is idempotent', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)

    controller.register()
    controller.dispose()
    controller.dispose()

    expect(globalShortcut.isRegistered(DEFAULT_HUD_SUMMON_SHORTCUT)).toBe(false)
    expect(controller.current()).toBeNull()
    expect(globalShortcut.unregister).toHaveBeenCalledTimes(1)
  })

  it('re-registering releases the old accelerator first', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)

    controller.register()
    controller.register()

    expect(globalShortcut.unregister).toHaveBeenCalledTimes(1)
    expect(globalShortcut.isRegistered(DEFAULT_HUD_SUMMON_SHORTCUT)).toBe(true)
  })
})
