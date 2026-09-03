import { describe, expect, it, vi } from 'vitest'

import {
  createHudAskShortcut,
  cropAroundCursor,
  DEFAULT_HUD_ASK_SHORTCUT,
  sanitizeHudPrefs,
  windowUnderCursor
} from './hud-ask'
import type { GlobalShortcutLike } from './quick-entry'

describe('sanitizeHudPrefs', () => {
  it('ships follow on, right-click off, and the default chord', () => {
    expect(sanitizeHudPrefs(undefined)).toEqual({
      follow: true,
      askShortcut: DEFAULT_HUD_ASK_SHORTCUT,
      askOnRightClick: false,
      pets: true,
      petByAgent: {}
    })
  })

  it('keeps only known pet choices, keyed by lower-cased profile', () => {
    expect(sanitizeHudPrefs({ petByAgent: { Gary: 'mina', jarvis: 'dragon', '': 'hank', warren: 3 } }).petByAgent).toEqual({
      gary: 'mina'
    })
    expect(sanitizeHudPrefs({ petByAgent: 'nope' }).petByAgent).toEqual({})
  })

  it('keeps a valid chord and drops an invalid one back to the default', () => {
    expect(sanitizeHudPrefs({ askShortcut: 'Control+Shift+K' }).askShortcut).toBe('Control+Shift+K')
    expect(sanitizeHudPrefs({ askShortcut: 'K' }).askShortcut).toBe(DEFAULT_HUD_ASK_SHORTCUT)
    expect(sanitizeHudPrefs({ askShortcut: 42 }).askShortcut).toBe(DEFAULT_HUD_ASK_SHORTCUT)
  })

  it('only honours literal booleans for the toggles', () => {
    expect(sanitizeHudPrefs({ follow: 'yes', askOnRightClick: 1 })).toMatchObject({
      follow: true,
      askOnRightClick: false
    })
    expect(sanitizeHudPrefs({ follow: false, askOnRightClick: true, pets: false })).toMatchObject({
      follow: false,
      askOnRightClick: true,
      pets: false
    })
  })
})

describe('cropAroundCursor', () => {
  const display = { x: 0, y: 0, width: 1920, height: 1080 }
  const image = { width: 1920, height: 1080 }
  const size = { width: 960, height: 600 }

  it('centres the crop on the cursor', () => {
    expect(cropAroundCursor({ x: 960, y: 540 }, display, image, size)).toEqual({ x: 480, y: 240, width: 960, height: 600 })
  })

  it('shifts rather than shrinks at the edges', () => {
    expect(cropAroundCursor({ x: 10, y: 10 }, display, image, size)).toEqual({ x: 0, y: 0, width: 960, height: 600 })
    expect(cropAroundCursor({ x: 1910, y: 1070 }, display, image, size)).toEqual({
      x: 960,
      y: 480,
      width: 960,
      height: 600
    })
  })

  it('maps the cursor proportionally when the thumbnail is smaller than the display', () => {
    const small = { width: 960, height: 540 }
    const crop = cropAroundCursor({ x: 960, y: 540 }, display, small, { width: 400, height: 300 })

    expect(crop).toEqual({ x: 280, y: 120, width: 400, height: 300 })
  })

  it('never asks for more than the image holds', () => {
    const crop = cropAroundCursor({ x: 100, y: 100 }, display, { width: 300, height: 200 }, size)

    expect(crop).toEqual({ x: 0, y: 0, width: 300, height: 200 })
  })

  it('handles a secondary display with its own origin', () => {
    const second = { x: 1920, y: 0, width: 1920, height: 1080 }

    expect(cropAroundCursor({ x: 2880, y: 540 }, second, image, size)).toEqual({
      x: 480,
      y: 240,
      width: 960,
      height: 600
    })
  })
})

describe('windowUnderCursor', () => {
  const win = (pid: number, app: string, x: number, y: number, width = 400, height = 300) => ({
    app,
    bounds: { x, y, width, height },
    id: pid,
    pid,
    title: `${app} window`
  })

  it('returns the frontmost window containing the cursor, skipping Hermes itself', () => {
    const windows = [win(1, 'Hermes', 0, 0, 2000, 2000), win(2, 'Figma', 100, 100), win(3, 'Chrome', 0, 0, 2000, 2000)]

    expect(windowUnderCursor(windows, { x: 150, y: 150 }, 1)?.app).toBe('Figma')
    expect(windowUnderCursor(windows, { x: 900, y: 900 }, 1)?.app).toBe('Chrome')
  })

  it('returns null when nothing but the desktop is under the cursor', () => {
    expect(windowUnderCursor([win(2, 'Figma', 100, 100)], { x: 5, y: 5 }, 1)).toBeNull()
    expect(windowUnderCursor([win(2, 'Empty', 0, 0, 0, 0)], { x: 0, y: 0 }, 1)).toBeNull()
  })
})

function fakeGlobalShortcut(taken: string[] = []) {
  const held = new Set(taken)
  const callbacks = new Map<string, () => void>()

  const globalShortcut: GlobalShortcutLike = {
    isRegistered: vi.fn((accelerator: string) => held.has(accelerator)),
    register: vi.fn((accelerator: string, callback: () => void) => {
      if (held.has(accelerator)) {
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

describe('createHudAskShortcut', () => {
  it('registers the chord and fires the ask on press', () => {
    const onAsk = vi.fn()
    const { globalShortcut, press } = fakeGlobalShortcut()
    const controller = createHudAskShortcut(globalShortcut, onAsk)

    expect(controller.register(DEFAULT_HUD_ASK_SHORTCUT)).toBe(true)
    expect(controller.current()).toBe(DEFAULT_HUD_ASK_SHORTCUT)
    press(DEFAULT_HUD_ASK_SHORTCUT)
    expect(onAsk).toHaveBeenCalledTimes(1)
  })

  it('releases the previous chord when re-registering, and reports a taken one', () => {
    const { globalShortcut, held } = fakeGlobalShortcut(['Control+Shift+K'])
    const controller = createHudAskShortcut(globalShortcut, vi.fn())

    controller.register(DEFAULT_HUD_ASK_SHORTCUT)
    expect(controller.register('Control+Shift+K')).toBe(false)
    expect(controller.current()).toBeNull()
    expect(held.has(DEFAULT_HUD_ASK_SHORTCUT)).toBe(false)
  })

  it('refuses a chord without a modifier without touching the OS', () => {
    const { globalShortcut } = fakeGlobalShortcut()
    const controller = createHudAskShortcut(globalShortcut, vi.fn())

    expect(controller.register('H')).toBe(false)
    expect(globalShortcut.register).not.toHaveBeenCalled()
  })

  it('dispose is idempotent', () => {
    const { globalShortcut, held } = fakeGlobalShortcut()
    const controller = createHudAskShortcut(globalShortcut, vi.fn())

    controller.register(DEFAULT_HUD_ASK_SHORTCUT)
    controller.dispose()
    controller.dispose()
    expect(held.size).toBe(0)
    expect(controller.current()).toBeNull()
  })
})
