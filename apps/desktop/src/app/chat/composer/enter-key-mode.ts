import { SLASH_COMMAND_RE } from '@hermes/shared'

export type ComposerEnterKeyIntent = 'native' | 'newline' | 'queue' | 'steer' | 'submit'

export interface ComposerEnterKeyIntentInput {
  busy?: boolean
  canSteer?: boolean
  enterSends: boolean
  key: string
  modKey?: boolean
  shiftKey?: boolean
}

/**
 * Resolve only the Enter-key mode decision. Higher-priority editor behaviors
 * (IME composition, completion popover acceptance, history navigation, etc.) run
 * before this helper in the composer keydown handler.
 */
export function resolveComposerEnterKeyIntent({
  busy = false,
  canSteer = false,
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

  return 'newline'
}

export interface LiveDraftSteerInput {
  attachmentCount: number
  /** Compacting, a blocking prompt, a queued-turn edit, or no steer handler. */
  blocked: boolean
  busy: boolean
  text: string
}

/** A live turn can take a steer only for a text-only, non-slash draft. */
export function canSteerLiveDraft({ attachmentCount, blocked, busy, text }: LiveDraftSteerInput): boolean {
  return busy && !blocked && attachmentCount === 0 && text.length > 0 && !SLASH_COMMAND_RE.test(text)
}
