import { formatCombo } from '@/lib/keybinds/combo'
import { activeSendGestures, type ComposerSendGesture, type ComposerSendPrefs } from '@/store/composer-prefs'

export interface SendHintWords {
  /** Gets the platform-correct chord, already formatted. */
  chord: (chord: string) => string
  /** 'tap Enter twice to send' */
  doubleTap: string
  /** 'Enter sends' — the default, worth confirming back after a switch. */
  enterSends: string
  /** 'hold Enter to send' */
  hold: string
  /** 'and it sends if you stop typing' — the auto-send, which has no key. */
  idle: string
  /** 'Enter starts a new line' — the base whenever the press breaks the line. */
  newline: string
  /** 'Enter after a pause sends' */
  pause: string
}

const GESTURE_WORDS: Record<ComposerSendGesture, keyof SendHintWords> = {
  doubleTap: 'doubleTap',
  hold: 'hold',
  pause: 'pause'
}

/** One sentence naming how the composer commits a draft for these prefs.
 *
 *  More than one gesture can be armed, so this reads as a list: the base case
 *  ("Enter starts a new line") followed by every way out that is switched on.
 *  The auto-send is appended even though it is not a key gesture — a draft still
 *  leaves the box that way, and the hint is where a user finds out it is on.
 *
 *  Shared by the composer placeholder (ambient, visible whenever the box is
 *  empty) and the settings toast, so the two cannot describe the same settings
 *  differently. */
export function composerSendHint(prefs: ComposerSendPrefs, words: SendHintWords): string {
  if (prefs.enterSends) {
    return words.enterSends
  }

  const phrases = activeSendGestures(prefs).map(gesture => words[GESTURE_WORDS[gesture]])

  if (prefs.sendOnIdle) {
    phrases.push(words.idle)
  }

  // Nothing but the press itself is armed, so name what still works.
  if (phrases.length === 0) {
    phrases.push(words.chord(formatCombo('mod+enter')))
  }

  // The base case is only worth naming while the press still does something:
  // with the line break switched off, a lone Enter does nothing at all and the
  // ways out are the whole sentence.
  return prefs.enterNewline ? `${words.newline} · ${phrases.join(' · ')}` : phrases.join(' · ')
}
