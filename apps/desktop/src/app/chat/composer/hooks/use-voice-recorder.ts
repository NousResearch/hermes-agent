import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { syncSttLease, VOICE_INPUT_LEASE } from '@/lib/stt-lease'
import { recordFeatureUse } from '@/store/desktop-metrics'
import { notify, notifyError } from '@/store/notifications'

import type { VoiceActivityState, VoiceStatus } from '../types'

import { useMicRecorder } from './use-mic-recorder'

/** How often the recorder is rotated so a complete, independently decodable
 *  blob is ready to transcribe while the user is still talking. 2s verified on
 *  this machine: local GPU STT returns a segment in ~0.3s hot, so the text
 *  trails the voice by well under a second. Confirmed working by dictation test.
 *  4s is also safe (more word context, slightly more lag) if accuracy ever
 *  matters more than immediacy — a shorter clip once dropped words at the seam. */
const STREAM_SEGMENT_MS = 2000

interface VoiceRecorderOptions {
  maxRecordingSeconds: number
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  focusInput: () => void
  onTranscript: (text: string) => void
  /** Set to false to keep the old one-shot behaviour (transcribe on stop). */
  streamPartial?: boolean
}

/** Join transcripts without doubling words at the seam: segments overlap in
 *  time, so a prefix of the new text that the previous one already ended with
 *  is dropped before concatenating. */
export function joinTranscriptParts(previous: string, next: string): string {
  const left = previous.trimEnd()
  const right = next.trim()

  if (!left) {
    return right
  }

  if (!right) {
    return left
  }

  const limit = Math.min(left.length, right.length)
  for (let size = limit; size > 0; size--) {
    if (left.slice(-size) === right.slice(0, size)) {
      return `${left}${right.slice(size)}`
    }
  }

  const needsSpace = !/\s$/.test(left) && !/^\s/.test(right) && /[\p{L}\p{N}]$/u.test(left)

  return `${left}${needsSpace ? ' ' : ''}${right}`
}

export function useVoiceRecorder({
  maxRecordingSeconds,
  onTranscribeAudio,
  focusInput,
  onTranscript,
  streamPartial = true
}: VoiceRecorderOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const { handle, level, recording } = useMicRecorder(voiceCopy)
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [partialText, setPartialText] = useState('')
  const startedAtRef = useRef(0)
  const intervalRef = useRef<number | null>(null)
  const timeoutRef = useRef<number | null>(null)
  /** Transcripts of the segments already delivered, in order. */
  const partsRef = useRef<string[]>([])
  /** Segments queued but not transcribed yet (transcription is serialised). */
  const queueRef = useRef<Blob[]>([])
  const drainingRef = useRef(false)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
    }
  }, [])

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

  const currentText = () => partsRef.current.reduce(joinTranscriptParts, '')

  /** Transcribe queued segments one at a time and show the running text. A
   *  serial queue matters: parallel requests would race and the slower one
   *  could land last, losing its words. */
  const drainQueue = async () => {
    if (drainingRef.current) {
      return
    }

    drainingRef.current = true

    try {
      while (queueRef.current.length) {
        const blob = queueRef.current.shift()

        if (!blob || !onTranscribeAudio) {
          continue
        }

        try {
          const text = (await onTranscribeAudio(blob)).trim()

          if (text) {
            partsRef.current.push(text)

            if (mountedRef.current) {
              setPartialText(currentText())
            }
          }
        } catch {
          // A failed segment must not abort the take: the final transcription
          // still runs over the whole audio, so nothing is lost.
        }
      }
    } finally {
      drainingRef.current = false
    }
  }

  const stop = async () => {
    clearTimers()
    const result = await handle.stop()

    if (!result) {
      setVoiceStatus('idle')
      setPartialText('')

      return
    }

    if (!onTranscribeAudio) {
      setVoiceStatus('idle')
      setPartialText('')

      return
    }

    setVoiceStatus('transcribing')

    try {
      // Wait for the in-flight segments, then transcribe the final slice. The
      // last blob is short and may cut a word, so the full-audio pass is what
      // the user keeps.
      await drainQueue()
      const tail = (await onTranscribeAudio(result.audio)).trim()
      const final = [currentText(), tail].reduce(joinTranscriptParts, '')

      if (!final) {
        notify({ kind: 'warning', title: voiceCopy.noSpeechDetected, message: voiceCopy.tryRecordingAgain })
      } else {
        onTranscript(final)
      }
    } catch (error) {
      notifyError(error, voiceCopy.transcriptionFailed)
    } finally {
      setVoiceStatus('idle')
      setPartialText('')
      partsRef.current = []
      queueRef.current = []
      // The transcript settled (or failed): this session no longer needs the
      // engine held. The backend keeps the shared model resident regardless.
      void syncSttLease(VOICE_INPUT_LEASE, false)
      focusInput()
    }
  }

  const start = async () => {
    if (!onTranscribeAudio) {
      notify({ kind: 'warning', title: voiceCopy.unavailable, message: voiceCopy.transcriptionUnavailable })

      return
    }

    partsRef.current = []
    queueRef.current = []
    drainingRef.current = false
    setPartialText('')

    try {
      await handle.start({
        onError: error => notifyError(error, voiceCopy.recordingFailed),
        ...(streamPartial
          ? {
              segmentMs: STREAM_SEGMENT_MS,
              onSegment: (audio: Blob) => {
                queueRef.current.push(audio)
                void drainQueue()
              }
            }
          : {})
      })
      // The mic is open, so a transcript is coming: warm the backend's STT
      // engine now so a cold local model loads while the user is still
      // speaking instead of inside the transcription timeout (#105955).
      // Fire-and-forget — recording must not wait on (or fail with) warm-up.
      void syncSttLease(VOICE_INPUT_LEASE, true)
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
    status: voiceStatus,
    partialText
  }

  return { dictate, voiceActivityState, voiceStatus }
}
