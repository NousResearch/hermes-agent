import { translateNow } from '@/i18n'
import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'
import {
  isVoicePlaybackActive,
  markVoicePlaybackInterrupted,
  stopVoicePlayback,
  takeVoicePlaybackInterrupted
} from '@/lib/voice-playback'
import { freezeComposerTransportPayload } from '@/store/composer'
import { notify } from '@/store/notifications'
import { consumePendingCredentialWarning, requestDesktopOnboarding } from '@/store/onboarding'

import type { SubmitTextOptions } from './utils'

/** Typed drafts freeze terminal chips before send; queue drains already froze
 * theirs. Confirmed plugin text is literal and cannot inherit live selections. */
export function prepareSubmitInput(
  rawText: string,
  options?: SubmitTextOptions
): { visibleText: string; bubbleOverride?: string } | null {
  let transportRaw = rawText
  let bubbleOverride = options?.displayText

  if (!options?.fromQueue && !options?.confirmedExternal) {
    const frozen = freezeComposerTransportPayload(rawText)

    if (frozen.missingLabels.length > 0) {
      notify({
        kind: 'warning',
        title: translateNow('composer.terminalSelectionMissingTitle'),
        message: translateNow('composer.terminalSelectionMissingBody')
      })

      return null
    }

    transportRaw = frozen.transportText

    if (!bubbleOverride && frozen.displayText !== frozen.transportText) {
      bubbleOverride = frozen.displayText
    }
  }

  return { visibleText: sanitizeComposerInput(transportRaw).trim(), bubbleOverride }
}

/** Ordinary user sends retain voice barge-in and deferred setup behavior.
 * Confirmed sends must not steal focus, consume voice state or open onboarding. */
export function prepareSubmitContext(options?: SubmitTextOptions): { interrupted: boolean } | null {
  if (options?.confirmedExternal) {
    return { interrupted: false }
  }

  if (isVoicePlaybackActive()) {
    markVoicePlaybackInterrupted()
    stopVoicePlayback()
  }

  if (!options?.fromQueue) {
    const deferredCredentialWarning = consumePendingCredentialWarning()

    if (deferredCredentialWarning) {
      requestDesktopOnboarding(deferredCredentialWarning)

      return null
    }
  }

  return { interrupted: takeVoicePlaybackInterrupted() }
}
