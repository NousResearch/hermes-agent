import { useCallback, useRef } from 'react'

import { interceptsTypedVoiceStop } from '@/lib/voice-stop-word'

import { runComposerMiddleware } from '../contrib'
import type { ChatBarProps } from '../types'

interface VoiceStopHandle {
  active: boolean
  end: () => void
}

export function useMiddlewareSubmit(onSubmitProp: ChatBarProps['onSubmit']) {
  // Typed stop phrase during an active voice conversation ends it — same
  // semantics as SAYING "stop" (voice-stop-word.ts) or clicking the pill's
  // end control. The caller populates it after useComposerVoice (the submit
  // wrapper is created first); render-time assignment keeps the ref current.
  const voiceStopRef = useRef<VoiceStopHandle>({ active: false, end: () => {} })

  // Every send (typed, queued, voice) passes through the contributed
  // middleware chain first — rewrite / pass-through / cancel. Empty chain =
  // exact pass-through, so surfaces without contributions are byte-identical.
  const onSubmit = useCallback<ChatBarProps['onSubmit']>(
    async (value, options) => {
      // Bare stop phrase typed while the voice conversation is live: end the
      // conversation (mic off, pill dismissed) instead of sending "stop" to
      // the agent. Spoken transcripts are already stop-checked inside
      // use-voice-conversation, so this only catches typed/queued sends.
      // Outside a voice conversation, typed "stop" is a normal message.
      const voiceStop = voiceStopRef.current

      if (interceptsTypedVoiceStop(voiceStop.active, value, options?.attachments?.length ?? 0)) {
        voiceStop.end()

        // Consumed (not rejected): report accepted so the submit engine
        // clears the draft instead of restoring "stop" into the composer.
        return true
      }

      const draft = await runComposerMiddleware({ text: value, attachments: options?.attachments })

      if (!draft) {
        return false
      }

      return onSubmitProp(draft.text, { ...options, attachments: draft.attachments })
    },
    [onSubmitProp]
  )

  return { onSubmit, voiceStopRef }
}
