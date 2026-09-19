import {
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_DEFAULT_MS,
  IDLE_SEND_DEFAULT_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  TYPING_IDLE_DEFAULT_MS
} from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const CONFIG_PATH = '/tmp/userData/composer-send.json'

/** Every field the prefs carry, so a new one can't be added without the tests
 *  noticing it changed shape. */
const DEFAULTS = {
  doubleEnterMs: DOUBLE_ENTER_DEFAULT_MS,
  enterNewline: true,
  enterSends: true,
  holdMs: HOLD_DEFAULT_MS,
  idleSendMs: IDLE_SEND_DEFAULT_MS,
  sendOnDoubleTap: false,
  sendOnHold: false,
  sendOnIdle: false,
  sendOnPause: false,
  sendGraceFor: SEND_GRACE_DEFAULT_REASONS,
  sendGraceMs: SEND_GRACE_DEFAULT_MS,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS
}

type DesktopWindow = { hermesDesktop?: unknown }

const setBridge = (bridge: unknown) => {
  ;(window as unknown as DesktopWindow).hermesDesktop = bridge
}

const loadStore = () => import('./composer-send')

describe('composer send preference', () => {
  beforeEach(() => {
    vi.resetModules()
    setBridge(undefined)
  })

  afterEach(() => {
    setBridge(undefined)
  })

  it('defaults to Enter sending, with every gesture off', async () => {
    const store = await loadStore()

    // The historical binding: an upgrade must not change what Enter does.
    expect(store.$composerSendPrefs.get()).toEqual(DEFAULTS)
    expect(store.activeSendGestures(DEFAULTS)).toEqual([])
  })

  it('arms only the gestures that were switched on, once Enter stops sending', async () => {
    const store = await loadStore()
    const both = { ...DEFAULTS, enterSends: false, sendOnHold: true, sendOnPause: true }

    expect(store.activeSendGestures(both)).toEqual(['pause', 'hold'])

    // Sending on the press leaves no room for a gesture, whatever the flags say.
    expect(store.activeSendGestures({ ...both, enterSends: true })).toEqual([])
  })

  it('mirrors what main has persisted, clamped', async () => {
    const get = vi.fn(async () => ({ doubleEnterMs: 99_999, enterSends: false, sendOnDoubleTap: true, path: CONFIG_PATH }))
    setBridge({ composerSend: { get, set: vi.fn() } })

    const store = await loadStore()
    const prefs = await store.refreshComposerSendPrefs()

    expect(prefs).toEqual({ ...DEFAULTS, doubleEnterMs: DOUBLE_ENTER_MAX_MS, enterSends: false, sendOnDoubleTap: true })
    expect(store.$composerSendConfigPath.get()).toBe(CONFIG_PATH)
  })

  it('keeps the line break switched off, which is a state of its own', async () => {
    const get = vi.fn(async () => ({ enterNewline: false, enterSends: false, path: CONFIG_PATH }))
    setBridge({ composerSend: { get, set: vi.fn() } })

    const store = await loadStore()
    const prefs = await store.refreshComposerSendPrefs()

    // Not coerced back on: a press that does nothing at all is the point of it.
    expect(prefs).toEqual({ ...DEFAULTS, enterNewline: false, enterSends: false })
  })

  it('writes through to main and keeps what main actually applied', async () => {
    const set = vi.fn(async (prefs: unknown) => ({
      ...DEFAULTS,
      doubleEnterMs: DOUBLE_ENTER_MIN_MS,
      path: CONFIG_PATH,
      requested: prefs
    }))

    setBridge({ composerSend: { get: vi.fn(), set } })

    const store = await loadStore()
    const applied = await store.setComposerSendPrefs({ doubleEnterMs: 5 })

    // Main clamps; the atom must follow main's answer, not the request.
    expect(set).toHaveBeenCalledWith({ ...DEFAULTS, doubleEnterMs: DOUBLE_ENTER_MIN_MS })
    expect(applied).toEqual({ ...DEFAULTS, doubleEnterMs: DOUBLE_ENTER_MIN_MS })
    expect(store.$composerSendPrefs.get()).toEqual({ ...DEFAULTS, doubleEnterMs: DOUBLE_ENTER_MIN_MS })
  })

  it('keeps the last known-good value when the write fails', async () => {
    const store = await loadStore()

    setBridge({
      composerSend: {
        get: vi.fn(),
        set: vi.fn(async () => {
          throw new Error('ipc down')
        })
      }
    })

    const applied = await store.setComposerSendPrefs({ enterSends: false })

    expect(applied).toEqual(DEFAULTS)
    expect(store.$composerSendPrefs.get()).toEqual(DEFAULTS)
  })

  it('stays usable on a surface with no desktop bridge (plain browser)', async () => {
    const store = await loadStore()
    const applied = await store.setComposerSendPrefs({ enterSends: false, sendOnIdle: true })

    expect(applied.enterSends).toBe(false)
    expect(applied.sendOnIdle).toBe(true)
    expect(store.$composerSendPrefs.get().sendOnIdle).toBe(true)
  })
})
