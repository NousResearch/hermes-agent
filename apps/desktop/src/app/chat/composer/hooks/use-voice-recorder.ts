import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

import type { VoiceActivityState, VoiceStatus } from '../types'

import { useMicRecorder } from './use-mic-recorder'

interface VoiceRecorderOptions {
  maxRecordingSeconds: number
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  focusInput: () => void
  onTranscript: (text: string) => void
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
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const startedAtRef = useRef(0)
  const operationGenerationRef = useRef(0)
  const intervalRef = useRef<number | null>(null)
  const timeoutRef = useRef<number | null>(null)

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
      operationGenerationRef.current += 1
      clearTimers()
    },
    []
  )

  const stop = async (autoSubmit = false): Promise<string | null> => {
    const generation = operationGenerationRef.current
    clearTimers()
    const result = await handle.stop()

    if (!result) {
      if (generation === operationGenerationRef.current) {
        setVoiceStatus('idle')
      }

      return null
    }

    if (!onTranscribeAudio) {
      if (generation === operationGenerationRef.current) {
        setVoiceStatus('idle')
      }

      return null
    }

    if (generation !== operationGenerationRef.current) {
      return null
    }

    setVoiceStatus('transcribing')

    try {
      const transcript = (await onTranscribeAudio(result.audio)).trim()

      if (generation !== operationGenerationRef.current) {
        return null
      }

      if (!transcript) {
        notify({ kind: 'warning', title: voiceCopy.noSpeechDetected, message: voiceCopy.tryRecordingAgain })
      } else if (autoSubmit) {
        // PTT submits through its composer seam after transcription resolves.
      } else {
        onTranscript(transcript)
      }

      return transcript
    } catch (error) {
      if (generation === operationGenerationRef.current) {
        notifyError(error, voiceCopy.transcriptionFailed)
      }

      return null
    } finally {
      if (generation === operationGenerationRef.current) {
        setVoiceStatus('idle')
        focusInput()
      }
    }
  }

  const start = async (withTimeout = true): Promise<boolean> => {
    if (!onTranscribeAudio) {
      notify({ kind: 'warning', title: voiceCopy.unavailable, message: voiceCopy.transcriptionUnavailable })

      return false
    }

    try {
      const started = await handle.start({ onError: error => notifyError(error, voiceCopy.recordingFailed) })

      if (!started) {
        setVoiceStatus('idle')

        return false
      }

      startedAtRef.current = Date.now()
      setElapsedSeconds(0)
      setVoiceStatus('recording')
      intervalRef.current = window.setInterval(() => setElapsedSeconds((Date.now() - startedAtRef.current) / 1000), 250)

      if (withTimeout) {
        const cap = Math.max(1, Math.min(Math.trunc(maxRecordingSeconds), 600))
        timeoutRef.current = window.setTimeout(() => void stop(), cap * 1000)
      }

      return true
    } catch (error) {
      setVoiceStatus('idle')
      notifyError(error, voiceCopy.recordingFailed)

      return false
    }
  }

  const dictate = () => {
    if (recording) {
      void stop()
    } else if (voiceStatus === 'idle') {
      void start()
    }
  }

  const startRecording = () => start(false)
  const stopRecording = () => stop(true)

  const cancelRecording = () => {
    operationGenerationRef.current += 1
    clearTimers()
    handle.cancel()
    setVoiceStatus('idle')
  }

  const voiceActivityState: VoiceActivityState = {
    elapsedSeconds,
    level,
    status: voiceStatus
  }

  return { dictate, startRecording, stopRecording, cancelRecording, voiceActivityState, voiceStatus }
}
