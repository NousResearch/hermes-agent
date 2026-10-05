/**
 * The shared composer-send contract: the defaults, the window clamps, and the
 * config round trip.
 *
 * It lives in the desktop workspace rather than beside the module, because
 * `apps/shared` declares no test runner: no vitest dependency, no `test` script,
 * and no vitest config in the repo collects `apps/shared/src/**`. A test kept
 * next to the module there would never execute. Importing through
 * `@hermes/shared` also proves the package surface exports what the app needs,
 * which is what caught `resetClampWarnings` missing from the index.
 */
import {
  activeSendGestures,
  clampHoldMs,
  clampIdleSendMs,
  composerConfigFromPrefs,
  composerPrefsFromConfig,
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_DEFAULT_MS,
  HOLD_MAX_MS,
  HOLD_MIN_MS,
  IDLE_SEND_DEFAULT_MS,
  IDLE_SEND_MAX_MS,
  IDLE_SEND_MIN_MS,
  normalizeComposerSendPrefs,
  resetClampWarnings,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  SEND_GRACE_MAX_MS,
  SEND_GRACE_MIN_MS,
  TYPING_IDLE_DEFAULT_MS,
  TYPING_IDLE_MIN_MS
} from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

/** Every field the prefs carry, so a new one cannot be added without a test
 *  noticing that the shape moved. */
const DEFAULTS = {
  commitOnPress: true,
  doubleEnterMs: DOUBLE_ENTER_DEFAULT_MS,
  enterNewline: false,
  enterSends: true,
  holdMs: HOLD_DEFAULT_MS,
  idleSendMs: IDLE_SEND_DEFAULT_MS,
  sendOnDoubleTap: true,
  sendOnHold: true,
  sendOnIdle: false,
  sendOnPause: false,
  sendGraceFor: SEND_GRACE_DEFAULT_REASONS,
  sendGraceMs: SEND_GRACE_DEFAULT_MS,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS
}

describe('composer send contract', () => {
  it('defaults to Enter sending, with the deliberate gestures already armed', () => {
    // The historical binding: an upgrade must not change what Enter does.
    expect(normalizeComposerSendPrefs({})).toEqual(DEFAULTS)
    expect(activeSendGestures(DEFAULTS)).toEqual([])
  })

  it('arms only the gestures that were switched on, once Enter stops sending', () => {
    const both = {
      ...DEFAULTS,
      enterSends: false,
      sendOnDoubleTap: false,
      sendOnHold: true,
      sendOnPause: true
    }

    expect(activeSendGestures(both)).toEqual(['pause', 'hold'])

    // Sending on the press leaves no room for a gesture, whatever the flags say.
    expect(activeSendGestures({ ...both, enterSends: true })).toEqual([])
  })

  it('reads the config record, clamping every window', () => {
    const prefs = composerPrefsFromConfig({
      double_enter_ms: 99_999,
      enter_newline: false,
      enter_sends: false,
      hold_ms: 1,
      idle_send_ms: 999_999,
      send_on_double_tap: true,
      typing_idle_ms: 0
    })

    expect(prefs).toEqual({
      ...DEFAULTS,
      doubleEnterMs: DOUBLE_ENTER_MAX_MS,
      enterNewline: false,
      enterSends: false,
      holdMs: HOLD_MIN_MS,
      idleSendMs: IDLE_SEND_MAX_MS,
      sendOnDoubleTap: true,
      typingIdleMs: TYPING_IDLE_MIN_MS
    })
  })

  it('keeps only the known grace reasons, in canonical order', () => {
    expect(composerPrefsFromConfig({ send_grace_for: ['hold', 'nonsense', 'enter', 'hold'] }).sendGraceFor).toEqual([
      'enter',
      'hold'
    ])

    // An empty list is "no delay", which is a choice, not an absent value.
    expect(composerPrefsFromConfig({ send_grace_for: [] }).sendGraceFor).toEqual([])
    expect(composerPrefsFromConfig({}).sendGraceFor).toEqual(SEND_GRACE_DEFAULT_REASONS)
  })

  it('clamps the grace window to its own bounds', () => {
    expect(composerPrefsFromConfig({ send_grace_ms: -5 }).sendGraceMs).toBe(SEND_GRACE_MIN_MS)
    expect(composerPrefsFromConfig({ send_grace_ms: 60_000 }).sendGraceMs).toBe(SEND_GRACE_MAX_MS)
  })

  it('ignores a shape no version of this setting ever wrote', () => {
    // The JSON file this setting once lived in had a `mode` enum. Config never
    // had it, so an unknown key must leave the defaults alone rather than being
    // guessed at — landing an old value on the default would turn Enter back
    // into a send for exactly the user who moved it away.
    expect(composerPrefsFromConfig({ mode: 'double-enter' })).toEqual(DEFAULTS)
  })

  it('round-trips prefs through the config record unchanged', () => {
    const prefs = {
      ...DEFAULTS,
      doubleEnterMs: 250,
      enterNewline: false,
      enterSends: false,
      holdMs: HOLD_MAX_MS,
      sendOnHold: true,
      sendOnIdle: true,
      sendGraceFor: ['pause', 'hold'] as const,
      sendGraceMs: SEND_GRACE_MAX_MS
    }

    expect(composerPrefsFromConfig(composerConfigFromPrefs(prefs))).toEqual(prefs)
  })

  it('normalizes a hand-edited config the same way the panel clamps a slider', () => {
    const handEdited = composerPrefsFromConfig({
      double_enter_ms: DOUBLE_ENTER_MIN_MS - 1,
      send_grace_for: 'hold'
    })

    expect(handEdited.doubleEnterMs).toBe(DOUBLE_ENTER_MIN_MS)
    // A scalar where a list belongs is not a reason, so the default stands.
    expect(handEdited.sendGraceFor).toEqual(SEND_GRACE_DEFAULT_REASONS)
    expect(handEdited.typingIdleMs).toBe(TYPING_IDLE_DEFAULT_MS)
  })
})

describe('clamp warnings', () => {
  /** Capture the warnings so a case can assert on both the text and the count. */
  function captureWarnings() {
    // Error level is what the desktop log captures, so that is the level asserted.
    const warn = vi.spyOn(console, 'error').mockImplementation(() => {})

    resetClampWarnings()

    return warn
  }

  it('names the key, the value and the bound when a hand-edited window is too large', () => {
    const warn = captureWarnings()

    expect(clampHoldMs(HOLD_MAX_MS + 1)).toBe(HOLD_MAX_MS)

    expect(warn).toHaveBeenCalledTimes(1)
    const line = String(warn.mock.calls[0][0])

    expect(line).toContain('desktop.composer.hold_ms')
    expect(line).toContain(String(HOLD_MAX_MS + 1))
    expect(line).toContain(`${HOLD_MIN_MS}-${HOLD_MAX_MS}`)

    warn.mockRestore()
  })

  it('names the key when the value is too small, and says what it used instead', () => {
    const warn = captureWarnings()

    expect(clampIdleSendMs(0)).toBe(IDLE_SEND_MIN_MS)

    expect(warn).toHaveBeenCalledTimes(1)
    const line = String(warn.mock.calls[0][0])

    expect(line).toContain('desktop.composer.idle_send_ms')
    expect(line).toContain(String(IDLE_SEND_MIN_MS))

    warn.mockRestore()
  })

  it('says a value that is not a number, rather than defaulting in silence', () => {
    const warn = captureWarnings()

    expect(clampHoldMs('soon')).toBe(HOLD_DEFAULT_MS)

    expect(warn).toHaveBeenCalledTimes(1)
    const line = String(warn.mock.calls[0][0])

    expect(line).toContain('is not a number')
    expect(line).toContain(String(HOLD_DEFAULT_MS))

    warn.mockRestore()
  })

  it('reports each key once, so a config refresh does not repeat the same line', () => {
    const warn = captureWarnings()

    clampHoldMs(HOLD_MAX_MS + 1)
    clampHoldMs(HOLD_MAX_MS + 1)
    clampIdleSendMs(0)

    // Two keys, two lines: one per key, not one per read.
    expect(warn).toHaveBeenCalledTimes(2)

    warn.mockRestore()
  })

  it('stays quiet for a value inside the bounds, and for a key that is absent', () => {
    const warn = captureWarnings()

    expect(clampHoldMs(HOLD_MIN_MS)).toBe(HOLD_MIN_MS)
    expect(clampHoldMs(HOLD_MAX_MS)).toBe(HOLD_MAX_MS)
    expect(clampHoldMs(undefined)).toBe(HOLD_DEFAULT_MS)
    expect(clampHoldMs(null)).toBe(HOLD_DEFAULT_MS)

    expect(warn).not.toHaveBeenCalled()

    warn.mockRestore()
  })
})
