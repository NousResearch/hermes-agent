import { describe, expect, it } from 'vitest'

import {
  activeSendGestures,
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
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  SEND_GRACE_MAX_MS,
  SEND_GRACE_MIN_MS,
  TYPING_IDLE_DEFAULT_MS,
  TYPING_IDLE_MAX_MS,
  TYPING_IDLE_MIN_MS
} from './composer-send'

/** Every field the prefs carry, so a new one cannot be added without a test
 *  noticing that the shape moved. */
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

describe('composer send contract', () => {
  it('defaults to Enter sending, with every gesture off', () => {
    // The historical binding: an upgrade must not change what Enter does.
    expect(normalizeComposerSendPrefs({})).toEqual(DEFAULTS)
    expect(activeSendGestures(DEFAULTS)).toEqual([])
  })

  it('arms only the gestures that were switched on, once Enter stops sending', () => {
    const both = { ...DEFAULTS, enterSends: false, sendOnHold: true, sendOnPause: true }

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
