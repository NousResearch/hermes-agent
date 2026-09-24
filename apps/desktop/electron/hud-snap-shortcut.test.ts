import { describe, expect, it, vi } from 'vitest'

import { createHudSnapShortcut, DEFAULT_HUD_SNAP_SHORTCUT } from './hud-snap-shortcut'
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

describe('createHudSnapShortcut', () => {
  it('registers CommandOrControl+Shift+G while the HUD is up', () => {
    const snap = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()

    const controller = createHudSnapShortcut(globalShortcut, snap)

    expect(controller.register()).toBe(true)
    expect(globalShortcut.isRegistered(DEFAULT_HUD_SNAP_SHORTCUT)).toBe(true)
  })

  it('dispose releases the accelerator', () => {
    const snap = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudSnapShortcut(globalShortcut, snap)

    controller.register()
    controller.dispose()

    expect(globalShortcut.isRegistered(DEFAULT_HUD_SNAP_SHORTCUT)).toBe(false)
  })

  it('register fails when the chord is already taken', () => {
    const snap = vi.fn<() => void>()
    const { globalShortcut } = fakeGlobalShortcut({ taken: [DEFAULT_HUD_SNAP_SHORTCUT] })
    const controller = createHudSnapShortcut(globalShortcut, snap)

    expect(controller.register()).toBe(false)
  })
})
