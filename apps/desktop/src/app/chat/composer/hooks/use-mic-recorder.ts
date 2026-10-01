import { useEffect, useRef, useState } from 'react'

import { closeMeterContext, meterContextsClosed } from '@/lib/mic-meter-context'

type BrowserAudioContext = typeof AudioContext

/** How long start() waits for a suspended meter context to reach 'running'.
 *  Past this the take records on, but its meter is unverified: a suspended
 *  analyser reads flat, so `heardSpeech=false` is no proof of silence. */
const METER_RESUME_TIMEOUT_MS = 300

/** Ignore over-threshold frames this long after capture opens (mic pop, start chime). */
export const SPEECH_SETTLE_MS = 300
/** Unbroken over-threshold time before a take counts as speech (filters clicks). */
export const SPEECH_ONSET_MS = 150

export interface MicRecorderOptions {
  onLevel?: (level: number) => void
  onError?: (error: Error) => void
  onSilence?: () => void
  /** The level meter died (AudioContext device/renderer error). Recording
   *  goes on, but silence detection and `heardSpeech` are blind from here. */
  onMeterFailure?: () => void
  silenceLevel?: number
  silenceMs?: number
  idleSilenceMs?: number
}

export interface MicRecording {
  audio: Blob
  durationMs: number
  heardSpeech: boolean
  /** The level meter failed during this take, so `heardSpeech` is unknown
   *  rather than false. */
  meterFailed?: boolean
  /** The meter's AudioContext was not running when capture began, so the
   *  opening of the take went unmetered and `heardSpeech` is unknown rather
   *  than false. Unlike `meterFailed`, the device is not considered broken. */
  meterUnverified?: boolean
}

export interface MicRecorderErrorCopy {
  microphoneAccessDenied: string
  microphoneConstraintsUnsupported: string
  microphoneInUse: string
  microphonePermissionDenied: string
  microphoneStartFailed: string
  microphoneUnsupported: string
  noMicrophone: string
}

interface MicRecorderHandle {
  start: (options?: MicRecorderOptions) => Promise<void>
  stop: () => Promise<MicRecording | null>
  cancel: () => void
}

/** Recorder + live-start mic failures → the same friendly copy: a DOMException
 *  name is mapped, an unrecognized DOMException falls back to the generic start
 *  copy, and anything else keeps its own message (non-mic failures must not be
 *  mislabeled as microphone problems). */
export function micError(error: unknown, copy: MicRecorderErrorCopy): Error {
  const name = error instanceof DOMException ? error.name : ''

  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return new Error(copy.microphonePermissionDenied)
  }

  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return new Error(copy.noMicrophone)
  }

  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return new Error(copy.microphoneInUse)
  }

  if (name === 'OverconstrainedError') {
    return new Error(copy.microphoneConstraintsUnsupported)
  }

  if (error instanceof DOMException) {
    return new Error(copy.microphoneStartFailed)
  }

  if (error instanceof Error) {
    return error
  }

  return new Error(copy.microphoneStartFailed)
}

export function useMicRecorder(copy: MicRecorderErrorCopy): {
  handle: MicRecorderHandle
  level: number
  recording: boolean
} {
  const [level, setLevel] = useState(0)
  const [recording, setRecording] = useState(false)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const audioContextRef = useRef<AudioContext | null>(null)
  const animationRef = useRef<number | null>(null)
  const startedAtRef = useRef(0)
  const heardSpeechRef = useRef(false)
  const meterFailedRef = useRef(false)
  const meterUnverifiedRef = useRef(false)
  const silenceTriggeredRef = useRef(false)
  const silenceStartedAtRef = useRef<number | null>(null)
  // Start of the current unbroken run of over-threshold frames (null = quiet).
  const loudSinceRef = useRef<number | null>(null)
  const stopResolverRef = useRef<((recording: MicRecording | null) => void) | null>(null)

  const cleanup = () => {
    if (animationRef.current) {
      window.cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    // Null the ref before closing so the context's own 'closed' statechange
    // isn't mistaken for a meter failure.
    const audioContext = audioContextRef.current
    audioContextRef.current = null
    closeMeterContext(audioContext)
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    recorderRef.current = null
    setLevel(0)
    setRecording(false)
    silenceTriggeredRef.current = false
  }

  useEffect(() => () => cleanup(), [])

  const startMeter = async (stream: MediaStream, options: MicRecorderOptions) => {
    const audioWindow = window as Window & { webkitAudioContext?: BrowserAudioContext }
    const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

    if (!AudioContextCtor) {
      return
    }

    const failMeter = () => {
      if (meterFailedRef.current || !recorderRef.current) {
        return
      }

      meterFailedRef.current = true

      if (animationRef.current) {
        window.cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }

      setLevel(0)
      // Deferred: a meter that fails while start() is still running must not
      // re-enter the caller before start() has resolved.
      window.setTimeout(() => {
        if (recorderRef.current) {
          options.onMeterFailure?.()
        }
      }, 0)
    }

    try {
      const audioContext = new AudioContextCtor()
      const analyser = audioContext.createAnalyser()
      const source = audioContext.createMediaStreamSource(stream)

      analyser.fftSize = 256
      const data = new Uint8Array(analyser.fftSize)

      source.connect(analyser)
      audioContextRef.current = audioContext

      // A device or renderer error kills the context without throwing
      // anywhere we'd see it; the analyser just goes flat. Watch for it.
      const failIfCurrent = () => {
        if (audioContextRef.current === audioContext) {
          failMeter()
        }
      }

      audioContext.addEventListener('error', failIfCurrent)
      audioContext.addEventListener('statechange', () => {
        if (audioContext.state === 'closed') {
          failIfCurrent()
        }
      })

      const tick = () => {
        analyser.getByteTimeDomainData(data)

        let sum = 0

        for (const value of data) {
          const centered = value - 128
          sum += centered * centered
        }

        const rms = Math.sqrt(sum / data.length)
        const normalized = Math.min(1, rms / 42)
        const now = Date.now()

        setLevel(normalized)
        options.onLevel?.(normalized)

        const speechThreshold = options.silenceLevel ?? 0
        const silenceMs = options.silenceMs ?? 0
        const idleSilenceMs = options.idleSilenceMs ?? 0

        if (speechThreshold > 0 && options.onSilence && !silenceTriggeredRef.current) {
          // A USB mic pops (and a start chime rings) the instant capture opens, and
          // a desk mic hears clicks: brief spikes over the threshold. Counting one
          // loud frame as speech started the end-of-utterance clock before the user
          // spoke, so the take ended as just the transient ("[clicking]") and the
          // real sentence was lost. Speech = sustained loudness, past the open settle.
          const loud = normalized >= speechThreshold && now - startedAtRef.current >= SPEECH_SETTLE_MS

          if (loud) {
            loudSinceRef.current ??= now
          } else {
            loudSinceRef.current = null
          }

          if (loud && (heardSpeechRef.current || now - loudSinceRef.current! >= SPEECH_ONSET_MS)) {
            heardSpeechRef.current = true
            silenceStartedAtRef.current = null
          } else if (loud) {
            // Onset still forming: neither speech yet nor silence.
          } else if (heardSpeechRef.current && silenceMs > 0) {
            silenceStartedAtRef.current ??= now

            if (now - silenceStartedAtRef.current >= silenceMs) {
              silenceTriggeredRef.current = true
              options.onSilence()

              return
            }
          } else if (!heardSpeechRef.current && idleSilenceMs > 0 && now - startedAtRef.current >= idleSilenceMs) {
            silenceTriggeredRef.current = true
            options.onSilence()

            return
          }
        }

        animationRef.current = window.requestAnimationFrame(tick)
      }

      tick()

      // Capture is already rolling; a suspended analyser reads flat, so speech
      // right after the mic opens would read as silence and the take would be
      // dropped unheard. Wait (bounded) for the context to run — past the
      // bound the take is kept, but flagged so STT judges it instead.
      if (audioContext.state !== 'running') {
        let timer: number | undefined

        await Promise.race([
          audioContext.resume().catch(failIfCurrent),
          new Promise<void>(resolve => {
            timer = window.setTimeout(resolve, METER_RESUME_TIMEOUT_MS)
          })
        ])
        window.clearTimeout(timer)

        if (audioContextRef.current === audioContext && (audioContext.state as AudioContextState) !== 'running') {
          meterUnverifiedRef.current = true
        }
      }
    } catch {
      failMeter()
    }
  }

  const start: MicRecorderHandle['start'] = async (options = {}) => {
    if (recorderRef.current) {
      return
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      throw new Error(copy.microphoneUnsupported)
    }

    const permitted = await window.hermesDesktop?.requestMicrophoneAccess?.()

    if (permitted === false) {
      throw new Error(copy.microphoneAccessDenied)
    }

    // The previous take's meter (or the barge monitor's) may still be
    // closing; opening another context on top of it is what trips the
    // AudioContext device error (#75329).
    await meterContextsClosed()

    let stream: MediaStream

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      })
    } catch (error) {
      throw micError(error, copy)
    }

    const mimeType =
      ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus', 'audio/ogg', 'audio/wav'].find(
        type => MediaRecorder.isTypeSupported(type)
      ) ?? ''

    let recorder: MediaRecorder

    try {
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    } catch (error) {
      stream.getTracks().forEach(track => track.stop())
      throw micError(error, copy)
    }

    chunksRef.current = []
    streamRef.current = stream
    recorderRef.current = recorder
    heardSpeechRef.current = false
    meterFailedRef.current = false
    meterUnverifiedRef.current = false
    silenceTriggeredRef.current = false
    silenceStartedAtRef.current = null
    loudSinceRef.current = null
    startedAtRef.current = Date.now()

    recorder.ondataavailable = event => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data)
      }
    }

    recorder.onstop = () => {
      const chunks = chunksRef.current
      const recordingType = recorder.mimeType || mimeType || 'audio/webm'
      const durationMs = Date.now() - startedAtRef.current
      const heardSpeech = heardSpeechRef.current
      const meterFailed = meterFailedRef.current
      const meterUnverified = meterUnverifiedRef.current

      chunksRef.current = []
      cleanup()

      const resolver = stopResolverRef.current
      stopResolverRef.current = null

      if (!chunks.length) {
        resolver?.(null)

        return
      }

      resolver?.({
        audio: new Blob(chunks, { type: recordingType }),
        durationMs,
        heardSpeech,
        meterFailed,
        meterUnverified
      })
    }

    recorder.onerror = event => {
      const error = micError((event as Event & { error?: unknown }).error, copy)
      const resolver = stopResolverRef.current
      stopResolverRef.current = null
      cleanup()
      options.onError?.(error)
      resolver?.(null)
    }

    recorder.start()
    setRecording(true)
    await startMeter(stream, options)
  }

  const stop: MicRecorderHandle['stop'] = () =>
    new Promise<MicRecording | null>(resolve => {
      const recorder = recorderRef.current

      if (!recorder || recorder.state === 'inactive') {
        cleanup()
        resolve(null)

        return
      }

      stopResolverRef.current = resolve
      recorder.stop()
    })

  const cancel: MicRecorderHandle['cancel'] = () => {
    const recorder = recorderRef.current
    const resolver = stopResolverRef.current
    stopResolverRef.current = null

    if (recorder && recorder.state !== 'inactive') {
      recorder.ondataavailable = null
      recorder.onerror = null
      recorder.onstop = null
      recorder.stop()
    }

    cleanup()
    resolver?.(null)
  }

  const handle: MicRecorderHandle = { start, stop, cancel }

  return { handle, level, recording }
}
