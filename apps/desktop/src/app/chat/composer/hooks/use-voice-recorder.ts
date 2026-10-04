import { useEffect, useRef, useState } from 'react'

import { type ResolvedOwner, resolveOwnerNow } from '@/hermes'
import { useI18n } from '@/i18n'
import { syncSttLease, VOICE_INPUT_LEASE } from '@/lib/stt-lease'
import { recordFeatureUse } from '@/store/desktop-metrics'
import { notify, notifyError } from '@/store/notifications'

import { useComposerScope } from '../scope'
import type { VoiceActivityState, VoiceStatus } from '../types'

import { useMicRecorder } from './use-mic-recorder'

interface VoiceRecorderOptions {
  maxRecordingSeconds: number
  onTranscribeAudio?: (audio: Blob, owner?: ResolvedOwner) => Promise<string>
  focusInput: () => void
  onTranscript: (text: string) => void
}

/**
 * One dictation, from mic-open to the settled transcript. Its owner is
 * resolved once at mic-open and used for the warm-up, the transcription and
 * the release, so a gateway/profile switch mid-recording cannot split them
 * across backends. `warmup` settles when that mic-open warm-up settled.
 */
interface Dictation {
  owner: ResolvedOwner
  warmup: Promise<void>
}

export function useVoiceRecorder({
  maxRecordingSeconds,
  onTranscribeAudio,
  focusInput,
  onTranscript
}: VoiceRecorderOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const { handle, level, recording } = useMicRecorder(voiceCopy)
  // The scope's session owner (a Bot tile's own connection + profile); the
  // main composer leaves it unset and the ambient scope fills it at mic-open.
  const { connectionId: ownerConnectionId, profile: ownerProfile } = useComposerScope()
  const ownerRef = useRef({ connectionId: ownerConnectionId, profile: ownerProfile })
  ownerRef.current = { connectionId: ownerConnectionId, profile: ownerProfile }
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const startedAtRef = useRef(0)
  const intervalRef = useRef<number | null>(null)
  const timeoutRef = useRef<number | null>(null)
  // The live dictation. stop() awaits its warm-up before transcribing (a cold
  // model load must not run inside the transcription request's decode
  // budget, #105955) and checks it is still the live one after every await:
  // unmount clears it, so a warm-up or transcript settling afterwards cannot
  // transcribe, insert text or move focus.
  const dictationRef = useRef<Dictation | null>(null)

  const clearTimers = () => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
  }

  useEffect(
    () => () => {
      clearTimers()
      const dictation = dictationRef.current
      dictationRef.current = null

      if (dictation) {
        void syncSttLease(VOICE_INPUT_LEASE, false, dictation.owner)
      }
    },
    []
  )

  const stop = async () => {
    clearTimers()
    const dictation = dictationRef.current
    const result = await handle.stop()
    const live = () => dictation !== null && dictationRef.current === dictation

    if (!dictation || !live()) {
      return // unmounted (or never started) while the recorder stopped
    }

    if (!result || !onTranscribeAudio) {
      finish(dictation)
      setVoiceStatus('idle')

      return
    }

    setVoiceStatus('transcribing')

    try {
      // Readiness first: let the mic-open warm-up settle before handing the
      // clip to transcription, so a cold model load no longer runs inside the
      // transcription request's timeout (#105955). syncSttLease never rejects,
      // and its budget is the lease request's own 180 s timeout.
      await dictation.warmup

      if (!live()) {
        return
      }

      const transcript = (await onTranscribeAudio(result.audio, dictation.owner)).trim()

      if (!live()) {
        return
      }

      if (!transcript) {
        notify({ kind: 'warning', title: voiceCopy.noSpeechDetected, message: voiceCopy.tryRecordingAgain })
      } else {
        onTranscript(transcript)
      }
    } catch (error) {
      if (live()) {
        notifyError(error, voiceCopy.transcriptionFailed)
      }
    } finally {
      if (live()) {
        finish(dictation)
        setVoiceStatus('idle')
        focusInput()
      }
    }
  }

  // The transcript settled (or failed): this dictation no longer needs the
  // engine held. Released to the owner that acquired it. The backend keeps the
  // shared model resident regardless.
  const finish = (dictation: Dictation) => {
    dictationRef.current = null
    void syncSttLease(VOICE_INPUT_LEASE, false, dictation.owner)
  }

  const start = async () => {
    if (!onTranscribeAudio) {
      notify({ kind: 'warning', title: voiceCopy.unavailable, message: voiceCopy.transcriptionUnavailable })

      return
    }

    try {
      await handle.start({ onError: error => notifyError(error, voiceCopy.recordingFailed) })
      // The mic is open, so a transcript is coming: warm the backend's STT
      // engine now so a cold local model loads while the user is still
      // speaking instead of inside the transcription timeout (#105955).
      // Fire-and-forget for the MIC UX — but keep the promise: stop() awaits
      // it as the readiness barrier before transcription (see above).
      const owner = resolveOwnerNow(ownerRef.current)
      dictationRef.current = { owner, warmup: syncSttLease(VOICE_INPUT_LEASE, true, owner) }
      startedAtRef.current = Date.now()
      setElapsedSeconds(0)
      setVoiceStatus('recording')
      recordFeatureUse('voice_dictation')
      intervalRef.current = window.setInterval(() => setElapsedSeconds((Date.now() - startedAtRef.current) / 1000), 250)
      const cap = Math.max(1, Math.min(Math.trunc(maxRecordingSeconds), 600))
      timeoutRef.current = window.setTimeout(() => void stop(), cap * 1000)
    } catch (error) {
      setVoiceStatus('idle')
      notifyError(error, voiceCopy.recordingFailed)
    }
  }

  const dictate = () => {
    if (recording) {
      void stop()
    } else if (voiceStatus === 'idle') {
      void start()
    }
  }

  const voiceActivityState: VoiceActivityState = {
    elapsedSeconds,
    level,
    status: voiceStatus
  }

  return { dictate, voiceActivityState, voiceStatus }
}
