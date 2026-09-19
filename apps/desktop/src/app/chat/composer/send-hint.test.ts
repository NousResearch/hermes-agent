import { normalizeComposerSendPrefs } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { formatCombo } from '@/lib/keybinds/combo'

import { composerSendHint } from './send-hint'

const words = {
  chord: (chord: string) => `${chord} sends`,
  doubleTap: 'tap Enter twice to send',
  enterSends: 'Enter sends',
  hold: 'hold Enter to send',
  idle: 'and it sends if you stop typing',
  newline: 'Enter starts a new line',
  pause: 'Enter after a pause sends'
}

/** Real defaults, so the hint is never tested against a shape the app cannot
 *  produce. */
const prefs = (over: Record<string, unknown> = {}) => normalizeComposerSendPrefs({ enterSends: false, ...over })

describe('composerSendHint', () => {
  it('confirms the default back, since that is all there is to say about it', () => {
    expect(composerSendHint(normalizeComposerSendPrefs({}), words)).toBe('Enter sends')
  })

  it('names the double tap when Enter only breaks lines', () => {
    expect(composerSendHint(prefs({ sendOnDoubleTap: true }), words)).toBe(
      'Enter starts a new line · tap Enter twice to send'
    )
  })

  it('lists every armed gesture, because more than one can be on', () => {
    const hint = composerSendHint(prefs({ sendOnDoubleTap: true, sendOnHold: true, sendOnPause: true }), words)

    expect(hint).toBe(
      'Enter starts a new line · tap Enter twice to send · Enter after a pause sends · hold Enter to send'
    )
  })

  it('mentions the auto-send even though it has no key of its own', () => {
    // The hint is where a user finds out a send can start on its own; nothing in
    // the composer would otherwise show it.
    expect(composerSendHint(prefs({ sendOnIdle: true }), words)).toBe(
      'Enter starts a new line · and it sends if you stop typing'
    )
  })

  it('prints the platform-correct chord when nothing else is armed', () => {
    const hint = composerSendHint(prefs(), words)

    // The contract is "hand the formatter's output through", not a literal
    // symbol — the env may not be the platform the app is running on. What
    // matters is that the raw `mod+enter` token never reaches the user.
    expect(hint).toBe(`Enter starts a new line · ${formatCombo('mod+enter')} sends`)
    expect(hint).not.toContain('mod+enter')
  })

  it('never mentions a gesture that is switched off', () => {
    const hint = composerSendHint(prefs({ sendOnDoubleTap: true }), words)

    expect(hint).not.toContain('hold')
    expect(hint).not.toContain('stop typing')
  })

  it('ignores armed gestures while Enter sends on the press', () => {
    // Stored, but unreachable: the press has already committed by then.
    const hint = composerSendHint(normalizeComposerSendPrefs({ enterSends: true, sendOnDoubleTap: true }), words)

    expect(hint).toBe('Enter sends')
  })
})
