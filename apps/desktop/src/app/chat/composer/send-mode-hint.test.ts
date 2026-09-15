import { describe, expect, it } from 'vitest'

import { formatCombo } from '@/lib/keybinds/combo'

import { composerSendModeHint } from './send-mode-hint'

const words = {
  chord: (chord: string) => `${chord} sends`,
  doubleTap: 'tap Enter twice to send',
  enterSends: 'Enter sends',
  newline: 'Enter starts a new line',
  hold: 'hold it to send',
  pause: 'sends once you stop typing'
}

describe('composerSendModeHint', () => {
  it('names the double tap when Enter only breaks lines', () => {
    expect(composerSendModeHint('double-enter', words)).toBe('Enter starts a new line · tap Enter twice to send')
  })

  it('prints the platform-correct chord for mod-enter', () => {
    const hint = composerSendModeHint('mod-enter', words)

    // The contract is "hand the formatter's output through", not a literal
    // symbol — the env may not be the platform the app is running on. What
    // matters is that the raw `mod+enter` token never reaches the user.
    expect(hint).toBe(`Enter starts a new line · ${formatCombo('mod+enter')} sends`)
    expect(hint).not.toContain('mod+enter')
  })

  it('confirms the default back after a switch away from it', () => {
    expect(composerSendModeHint('enter', words)).toBe('Enter sends')
  })

  it('says hold, not double tap, when the send is the long press', () => {
    expect(composerSendModeHint('hold', words)).toBe('Enter starts a new line · hold it to send')
  })
})
