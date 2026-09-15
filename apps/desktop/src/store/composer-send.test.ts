import {
  COMPOSER_SEND_DEFAULT_MODE,
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_DEFAULT_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_SCOPE,
  TYPING_IDLE_DEFAULT_MS
} from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const CONFIG_PATH = '/tmp/userData/composer-send.json'

/** Every field the prefs carry, so a new one can't be added without the tests
 *  noticing it changed shape. */
const DEFAULTS = {
  mode: COMPOSER_SEND_DEFAULT_MODE,
  doubleEnterMs: DOUBLE_ENTER_DEFAULT_MS,
  holdMs: HOLD_DEFAULT_MS,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS,
  sendGrace: SEND_GRACE_DEFAULT_SCOPE,
  sendGraceMs: SEND_GRACE_DEFAULT_MS
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

  it('defaults to Enter sends with the shipped double-tap window', async () => {
    const store = await loadStore()

    expect(store.$composerSendPrefs.get()).toEqual(DEFAULTS)
    expect(store.enterBreaksLine(COMPOSER_SEND_DEFAULT_MODE)).toBe(false)
    expect(store.enterBreaksLine('pause')).toBe(true)
  })

  it('mirrors what main has persisted, clamped', async () => {
    const get = vi.fn(async () => ({ doubleEnterMs: 99_999, mode: 'double-enter', path: CONFIG_PATH }))
    setBridge({ composerSend: { get, set: vi.fn() } })

    const store = await loadStore()
    const prefs = await store.refreshComposerSendPrefs()

    expect(prefs).toEqual({ ...DEFAULTS, mode: 'double-enter', doubleEnterMs: DOUBLE_ENTER_MAX_MS })
    expect(store.$composerSendMode.get()).toBe('double-enter')
    expect(store.$composerSendConfigPath.get()).toBe(CONFIG_PATH)
    expect(store.enterBreaksLine('double-enter')).toBe(true)
  })

  it('writes through to main and keeps what main actually applied', async () => {
    const set = vi.fn(async (prefs: unknown) => ({
      ...DEFAULTS,
      doubleEnterMs: DOUBLE_ENTER_MIN_MS,
      mode: 'mod-enter',
      path: CONFIG_PATH,
      requested: prefs
    }))

    setBridge({ composerSend: { get: vi.fn(), set } })

    const store = await loadStore()
    const applied = await store.setComposerSendPrefs({ mode: 'mod-enter', doubleEnterMs: 5 })

    // Main clamps; the atom must follow main's answer, not the request.
    expect(set).toHaveBeenCalledWith({ ...DEFAULTS, mode: 'mod-enter', doubleEnterMs: DOUBLE_ENTER_MIN_MS })
    expect(applied).toEqual({ ...DEFAULTS, mode: 'mod-enter', doubleEnterMs: DOUBLE_ENTER_MIN_MS })
    expect(store.$composerSendPrefs.get()).toEqual({ ...DEFAULTS, mode: 'mod-enter', doubleEnterMs: DOUBLE_ENTER_MIN_MS })
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

    const applied = await store.setComposerSendPrefs({ mode: 'double-enter' })

    expect(applied).toEqual(DEFAULTS)
    expect(store.$composerSendPrefs.get()).toEqual(DEFAULTS)
  })

  it('stays usable on a surface with no desktop bridge (plain browser)', async () => {
    const store = await loadStore()
    const applied = await store.setComposerSendPrefs({ mode: 'mod-enter' })

    expect(applied.mode).toBe('mod-enter')
    expect(store.$composerSendMode.get()).toBe('mod-enter')
  })
})
