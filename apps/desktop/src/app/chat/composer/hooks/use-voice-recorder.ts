import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { sttStreamingRelayAvailable } from '@/lib/voice-client-direct'
import { type DictationStreamSession, openDictationStream } from '@/lib/voice-stream'
import { notify, notifyError } from '@/store/notifications'

import type { VoiceActivityState, VoiceStatus } from '../types'

import { useMicRecorder } from './use-mic-recorder'

interface VoiceRecorderOptions {
  maxRecordingSeconds: number
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  focusInput: () => void
  onTranscript: (text: string) => void
}

/**
 * Push-to-talk dictation (mic icon in the composer): record → transcript.
 *
 * When the active STT provider supports live streaming (see
 * `sttStreamingRelayAvailable`), PCM captured WHILE the user speaks is
 * streamed to the host's /api/audio/transcribe-stream relay, so the
 * transcript is ready right at end-of-recording instead of after a post-hoc
 * file transcription. The MediaRecorder blob is still produced in parallel
 * and used as the fallback whenever the streaming leg fails — dictation
 * keeps working exactly as before on older backends or non-streaming STT.
 */
/** Bound a promise so a silent peer can never freeze the UI indefinitely. */
async function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), ms)
      })
    ])
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer)
    }
  }
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
  const intervalRef = useRef<number | null>(null)
  const timeoutRef = useRef<number | null>(null)
  const recordingRef = useRef(false)
  const streamSessionRef = useRef<DictationStreamSession | null>(null)
  const pcmBufferRef = useRef<ArrayBuffer[]>([])

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

  useEffect(() => () => clearTimers(), [])

  /** Feed live PCM to the streaming session; buffer briefly until it opens. */
  const onPcmChunk = (chunk: ArrayBuffer) => {
    const session = streamSessionRef.current

    if (session) {
      session.pushAudio(chunk)

      return
    }

    const buffer = pcmBufferRef.current

    // ~64 s of 16 kHz s16le at one ~256 ms chunk per push.
    if (buffer.length < 250) {
      buffer.push(chunk)
    }
  }

  const openStreamSession = async () => {
    const session = await openDictationStream()

    if (!session) {
      return
    }

    if (!recordingRef.current) {
      session.cancel()

      return
    }

    streamSessionRef.current = session
    const buffered = pcmBufferRef.current
    pcmBufferRef.current = []

    for (const chunk of buffered) {
      session.pushAudio(chunk)
    }
  }

  /** Transcribe a finished recording blob (the non-streaming path). */
  const transcribeBlob = async (audio: Blob) => {
    if (!onTranscribeAudio) {
      setVoiceStatus('idle')

      return
    }

    setVoiceStatus('transcribing')

    try {
      const transcript = (await onTranscribeAudio(audio)).trim()

      if (!transcript) {
        notify({ kind: 'warning', title: voiceCopy.noSpeechDetected, message: voiceCopy.tryRecordingAgain })
      } else {
        onTranscript(transcript)
      }
    } catch (error) {
      notifyError(error, voiceCopy.transcriptionFailed)
    } finally {
      setVoiceStatus('idle')
      focusInput()
    }
  }

  const stop = async () => {
    clearTimers()
    recordingRef.current = false

    const session = streamSessionRef.current
    streamSessionRef.current = null
    pcmBufferRef.current = []

    const result = await handle.stop()

    if (!result) {
      session?.cancel()
      setVoiceStatus('idle')

      return
    }

    if (session) {
      // Streaming leg: transcript should already be (almost) ready. Bound it
      // hard — a silent provider/socket must never leave the mic button
      // frozen in the recording state; on any failure we fall back to the
      // recorded blob, which is exactly today's non-streaming behaviour.
      setVoiceStatus('transcribing')
      const startedAt = Date.now()

      try {
        const transcript = (await withTimeout(session.stop(), 25_000, 'live STT timed out'))
          ?.trim() ?? ''

        if (transcript) {
          console.info(`[dictation] streaming OK in ${Date.now() - startedAt}ms`)
          onTranscript(transcript)
          setVoiceStatus('idle')
          focusInput()

          return
        }
      } catch (error) {
        // Fall through to the recorded-blob path. Close the live socket so a
        // silent provider can't leak a half-open session into the next take.
        const reason = error instanceof Error ? error.message : String(error)
        console.warn(`[dictation] streaming failed (${reason}); falling back to blob relay`)
        notifyError(
          new Error(`${reason} — retrying with the standard transcription`),
          voiceCopy.transcriptionFailed
        )
        session.cancel()
      }
    }

    await transcribeBlob(result.audio)
  }

  const start = async () => {
    if (!onTranscribeAudio) {
      notify({ kind: 'warning', title: voiceCopy.unavailable, message: voiceCopy.transcriptionUnavailable })

      return
    }

    // Streaming is a relay capability: only when the backend's STT provider
    // reports streaming support AND we still have a mic blob as fallback.
    let streaming = false

    try {
      streaming = await sttStreamingRelayAvailable()
    } catch {
      streaming = false
    }

    console.info(`[dictation] mode=${streaming ? 'streaming' : 'blob'} (stt.streaming=${streaming})`)

    if (streaming) {
      pcmBufferRef.current = []
    }

    try {
      await handle.start({
        onError: error => notifyError(error, voiceCopy.recordingFailed),
        onPcm: streaming ? onPcmChunk : undefined
      })
    } catch (error) {
      setVoiceStatus('idle')
      notifyError(error, voiceCopy.recordingFailed)

      return
    }

    recordingRef.current = true
    startedAtRef.current = Date.now()
    setElapsedSeconds(0)
    setVoiceStatus('recording')
    intervalRef.current = window.setInterval(() => setElapsedSeconds((Date.now() - startedAtRef.current) / 1000), 250)
    const cap = Math.max(1, Math.min(Math.trunc(maxRecordingSeconds), 600))
    timeoutRef.current = window.setTimeout(() => void stop(), cap * 1000)

    if (streaming) {
      void openStreamSession()
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
