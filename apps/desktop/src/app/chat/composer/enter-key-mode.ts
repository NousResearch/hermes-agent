export type ComposerEnterKeyIntent = 'native' | 'newline' | 'queue' | 'steer' | 'submit' | 'ignore'

export interface ComposerEnterKeyIntentInput {
  busy?: boolean
  canSteer?: boolean
  /** What a press does when it is not sending: break the line, or nothing at
   *  all. Only consulted while the gate is closed. Defaults to the line break,
   *  which is what an install that never opened Settings expects. */
  enterNewline?: boolean
  enterSends: boolean
  key: string
  modKey?: boolean
  shiftKey?: boolean
}

/**
 * Resolve only the Enter-key mode decision. Higher-priority editor behaviors
 * (IME composition, completion popover acceptance, history navigation, etc.) run
 * before this helper in the composer keydown handler, and so do the send
 * gestures — a press a gesture claims never reaches here.
 *
 * `ignore` is the one outcome with no keystroke in it: the settings said a bare
 * Enter neither sends nor breaks the line, so the press is swallowed rather than
 * left to the editor.
 */
export function resolveComposerEnterKeyIntent({
  busy = false,
  canSteer = false,
  enterNewline = true,
  enterSends,
  key,
  modKey = false,
  shiftKey = false
}: ComposerEnterKeyIntentInput): ComposerEnterKeyIntent {
  if (key !== 'Enter') {
    return 'native'
  }

  if (enterSends) {
    if (modKey && !shiftKey) {
      return busy ? 'queue' : 'submit'
    }

    return shiftKey ? 'native' : 'submit'
  }

  if (modKey && !shiftKey) {
    return busy ? 'queue' : 'submit'
  }

  if (shiftKey) {
    return canSteer ? 'steer' : 'newline'
  }

  return enterNewline ? 'newline' : 'ignore'
}
