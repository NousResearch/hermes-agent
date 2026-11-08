import { DOUBLE_ENTER_MAX_MS, normalizeComposerSendPrefs } from '@hermes/shared'
import { afterEach, describe, expect, it } from 'vitest'

import { $composerSendPrefs, applyComposerPrefsFromConfig } from './composer-prefs'

const DEFAULTS = normalizeComposerSendPrefs({})

afterEach(() => $composerSendPrefs.set(DEFAULTS))

describe('composer prefs from Hermes config', () => {
  it('mirrors the config record into the atom the composer reads', () => {
    applyComposerPrefsFromConfig({
      desktop: { composer: { double_enter_ms: 250, enter_sends: false, send_on_double_tap: true } }
    })

    const prefs = $composerSendPrefs.get()

    expect(prefs.enterSends).toBe(false)
    expect(prefs.sendOnDoubleTap).toBe(true)
    expect(prefs.doubleEnterMs).toBe(250)
  })

  it('preserves the backward-compatible Enter-to-send default for missing or invalid values', () => {
    $composerSendPrefs.set({ ...DEFAULTS, enterSends: false })
    applyComposerPrefsFromConfig({ desktop: { composer: { enter_sends: 'false' } } })

    expect($composerSendPrefs.get().enterSends).toBe(true)

    $composerSendPrefs.set({ ...DEFAULTS, enterSends: false })
    applyComposerPrefsFromConfig({})

    expect($composerSendPrefs.get().enterSends).toBe(true)
  })

  it('clamps a hand-edited value to the bounds the panel writes', () => {
    applyComposerPrefsFromConfig({ desktop: { composer: { double_enter_ms: 99_999 } } })

    expect($composerSendPrefs.get().doubleEnterMs).toBe(DOUBLE_ENTER_MAX_MS)
  })

  it('reads the whole contract, not just the gate, from one load', () => {
    applyComposerPrefsFromConfig({
      desktop: { composer: { enter_newline: false, send_grace_for: ['hold'], send_on_idle: true } }
    })

    const prefs = $composerSendPrefs.get()

    expect(prefs.enterNewline).toBe(false)
    expect(prefs.sendOnIdle).toBe(true)
    expect(prefs.sendGraceFor).toEqual(['hold'])
  })
})
