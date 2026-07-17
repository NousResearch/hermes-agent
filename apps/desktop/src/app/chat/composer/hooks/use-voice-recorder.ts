import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

import type { VoiceActivityState, VoiceStatus } from '../types'

import { useMicRecorder } from './use-mic-recorder'

interface VoiceRecorderOptions {
  maxRecordingSeconds: number
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  focusInput: () => void
  onIdle?: () => void
  onTranscript: (text: string, submit: boolean) => void
}

type RecorderPhase = 'cancelling' | 'idle' | 'recording' | 'starting' | 'stopping' | 'transcribing'

export function useVoiceRecorder({
  maxRecordingSeconds,
  onTranscribeAudio,
  focusInput,
  onIdle,
  onTranscript
}: VoiceRecorderOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const { handle, level } = useMicRecorder(voiceCopy)
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const startedAtRef = useRef(0)
  const intervalRef = useRef<number | null>(null)
  const timeoutRef = useRef<number | null>(null)
  const phaseRef = useRef<RecorderPhase>('idle')
  const generationRef = useRef(0)
  const pendingStopRef = useRef<boolean | null>(null)
  const mountedRef = useRef(true)
  const handleRef = useRef(handle)
  const callbacksRef = useRef({ focusInput, onIdle, onTranscript, onTranscribeAudio, voiceCopy })
  handleRef.current = handle
  callbacksRef.current = { focusInput, onIdle, onTranscript, onTranscribeAudio, voiceCopy }
  const cancellationPending = () => phaseRef.current === 'cancelling'

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

  const finish = () => {
    phaseRef.current = 'idle'
    pendingStopRef.current = null

    if (mountedRef.current) {
      setVoiceStatus('idle')
    }

    callbacksRef.current.onIdle?.()
  }

  useEffect(
    () => {
      mountedRef.current = true

      return () => {
        mountedRef.current = false
        generationRef.current += 1
        phaseRef.current = 'idle'
        pendingStopRef.current = null
        clearTimers()
        handleRef.current.cancel()
      }
    },
    []
  )

  const stop = async (submitTranscript = false) => {
    if (phaseRef.current === 'starting') {
      pendingStopRef.current = (pendingStopRef.current ?? false) || submitTranscript

      return
    }

    if (phaseRef.current !== 'recording') {
      return
    }

    phaseRef.current = 'stopping'
    clearTimers()
    const generation = generationRef.current
    const result = await handle.stop()

    if (generation !== generationRef.current) {
      return
    }

    if (!result) {
      finish()

      return
    }

    const transcribe = callbacksRef.current.onTranscribeAudio

    if (!transcribe) {
      finish()

      return
    }

    phaseRef.current = 'transcribing'
    setVoiceStatus('transcribing')

    try {
      const transcript = (await transcribe(result.audio)).trim()

      if (generation !== generationRef.current) {
        return
      }

      if (!transcript) {
        const copy = callbacksRef.current.voiceCopy

        notify({ kind: 'warning', title: copy.noSpeechDetected, message: copy.tryRecordingAgain })
      } else {
        callbacksRef.current.onTranscript(transcript, submitTranscript)
      }
    } catch (error) {
      if (generation === generationRef.current) {
        notifyError(error, callbacksRef.current.voiceCopy.transcriptionFailed)
      }
    } finally {
      if (generation === generationRef.current) {
        finish()
        callbacksRef.current.focusInput()
      }
    }
  }

  const start = async () => {
    if (phaseRef.current !== 'idle') {
      return
    }

    if (!callbacksRef.current.onTranscribeAudio) {
      const copy = callbacksRef.current.voiceCopy

      notify({ kind: 'warning', title: copy.unavailable, message: copy.transcriptionUnavailable })
      finish()

      return
    }

    const generation = ++generationRef.current
    phaseRef.current = 'starting'
    setVoiceStatus('recording')

    try {
      await handle.start({
        onError: error => {
          if (generation !== generationRef.current) {
            return
          }

          notifyError(error, callbacksRef.current.voiceCopy.recordingFailed)
          generationRef.current += 1
          clearTimers()
          finish()
        }
      })

      if (generation !== generationRef.current) {
        handle.cancel()

        if (cancellationPending()) {
          finish()
        }

        return
      }

      phaseRef.current = 'recording'
      startedAtRef.current = Date.now()
      setElapsedSeconds(0)
      intervalRef.current = window.setInterval(() => setElapsedSeconds((Date.now() - startedAtRef.current) / 1000), 250)
      const cap = Math.max(1, Math.min(Math.trunc(maxRecordingSeconds), 600))
      timeoutRef.current = window.setTimeout(() => void stop(), cap * 1000)
      const pendingStop = pendingStopRef.current

      if (pendingStop !== null) {
        pendingStopRef.current = null
        void stop(pendingStop)
      }
    } catch (error) {
      if (generation !== generationRef.current) {
        if (cancellationPending()) {
          finish()
        }

        return
      }

      notifyError(error, callbacksRef.current.voiceCopy.recordingFailed)
      finish()
    }
  }

  const dictate = (submitTranscript = false) => {
    if (phaseRef.current === 'idle') {
      void start()
    } else if (phaseRef.current === 'starting') {
      pendingStopRef.current = (pendingStopRef.current ?? false) || submitTranscript
    } else if (phaseRef.current === 'recording') {
      void stop(submitTranscript)
    }
  }

  const cancel = () => {
    const phase = phaseRef.current

    if (phase === 'idle' || phase === 'cancelling') {
      return
    }

    clearTimers()
    pendingStopRef.current = null
    generationRef.current += 1
    handle.cancel()

    if (phase === 'starting') {
      // The permission/getUserMedia promise cannot be aborted. Keep ownership
      // until it settles, then immediately tear down the stream in start().
      phaseRef.current = 'cancelling'

      return
    }

    finish()
  }

  const voiceActivityState: VoiceActivityState = {
    elapsedSeconds,
    level,
    status: voiceStatus
  }

  return { cancel, dictate, voiceActivityState, voiceStatus }
}
