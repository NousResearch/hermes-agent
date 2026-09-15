import { formatCombo } from '@/lib/keybinds/combo'
import type { ComposerSendMode } from '@/store/composer-send'

export interface SendModeHintWords {
  /** 'Enter sends' — the default, worth confirming back after a switch. */
  enterSends: string
  newline: string
  doubleTap: string
  /** 'sends once you stop typing' — the pause mode's inferred send. */
  pause: string
  /** Gets the platform-correct chord, already formatted. */
  chord: (chord: string) => string
}

/** One sentence naming how the composer commits a draft in `mode`.
 *
 *  Shared by the composer placeholder (ambient, always visible when the box is
 *  empty) and the settings toast (the moment the mode changes), so the two
 *  surfaces cannot end up describing the same mode differently. */
export function composerSendModeHint(mode: ComposerSendMode, words: SendModeHintWords): string {
  if (mode === 'double-enter') {
    return `${words.newline} · ${words.doubleTap}`
  }

  if (mode === 'pause') {
    return `${words.newline} · ${words.pause}`
  }

  if (mode === 'mod-enter') {
    return `${words.newline} · ${words.chord(formatCombo('mod+enter'))}`
  }

  return words.enterSends
}
