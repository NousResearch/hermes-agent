import { describe, expect, it, vi } from 'vitest'

import { createHudSummonShortcut, DEFAULT_HUD_SUMMON_SHORTCUT } from './hud-summon-shortcut'
import type { GlobalShortcutLike } from './quick-entry'

function fakeGlobalShortcut(options: { register?: boolean; taken?: string[] } = {}) {
  const held = new Set(options.taken ?? [])

  const globalShortcut: GlobalShortcutLike = {
    isRegistered: vi.fn((accelerator: string) => held.has(accelerator)),
    register: vi.fn((accelerator: string, _callback: () => void) => {
      if (options.register === false || held.has(accelerator)) {
        return false
      }

      held.add(accelerator)

      return true
    }),
    unregister: vi.fn((accelerator: string) => void held.delete(accelerator))
  }

  return { globalShortcut, held }
}

describe('createHudSummonShortcut', () => {
  it('registers CommandOrControl+Shift+U while enabled', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)
    expect(controller.register()).toBe(true)
    expect(globalShortcut.isRegistered(DEFAULT_HUD_SUMMON_SHORTCUT)).toBe(true)
  })

  it('dispose releases the accelerator', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)
    controller.register()
    controller.dispose()
    expect(globalShortcut.isRegistered(DEFAULT_HUD_SUMMON_SHORTCUT)).toBe(false)
  })

  it('register fails when the chord is already taken', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut({ taken: [DEFAULT_HUD_SUMMON_SHORTCUT] })
    const controller = createHudSummonShortcut(globalShortcut, summon)
    expect(controller.register()).toBe(false)
  })

  it('register re-arms after dispose (re-register works)', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)
    expect(controller.register()).toBe(true)
    controller.dispose()
    expect(controller.register()).toBe(true)
    expect(globalShortcut.isRegistered(DEFAULT_HUD_SUMMON_SHORTCUT)).toBe(true)
  })

  it('invokes the onSummon callback when the accelerator fires', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)
    controller.register()

    const registeredCallback = vi.mocked(globalShortcut.register).mock.calls[0][1]
    registeredCallback()

    expect(summon).toHaveBeenCalledOnce()
  })

  it('dispose is idempotent (repeated calls do not throw)', () => {
    const summon = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSummonShortcut(globalShortcut, summon)
    controller.register()

    controller.dispose()
    controller.dispose()

    expect(globalShortcut.isRegistered(DEFAULT_HUD_SUMMON_SHORTCUT)).toBe(false)
  })
})
