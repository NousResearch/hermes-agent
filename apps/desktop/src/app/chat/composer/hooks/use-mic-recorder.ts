import { useEffect, useRef, useState } from 'react'

import { hostHas } from '@/lib/host-capabilities'
import { closeMeterContext, meterContextsClosed } from '@/lib/mic-meter-context'
import { startVoiceLevelMeter } from '@/lib/voice-level-meter'

type BrowserAudioContext = typeof AudioContext

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
  /** Live PCM tap (streaming dictation): mono s16le chunks at the meter context's rate, which
   *  `onPcmRate` reports once before the first chunk. The MediaRecorder blob is unaffected. */
  onPcm?: (chunk: ArrayBuffer) => void
  onPcmRate?: (sampleRate: number) => void
}

export interface MicRecording {
  audio: Blob
  durationMs: number
  heardSpeech: boolean
  /** The level meter failed during this take, so `heardSpeech` is unknown
   *  rather than false. */
  meterFailed?: boolean
}

export interface MicRecorderErrorCopy {
  microphoneAccessDenied: string
  microphoneConstraintsUnsupported: string
  microphoneInUse: string
  microphoneNeedsHttps: string
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

/** "Microphone unsupported" copy. The Webapp opened over plain HTTP from another machine is an
 *  insecure context with no mediaDevices: it says HTTPS or a tunnel fixes that. Native Desktop
 *  keeps the plain copy. */
function unsupportedMicrophone(copy: MicRecorderErrorCopy): string {
  return hostHas('webappCopy') && window.isSecureContext === false
    ? copy.microphoneNeedsHttps
    : copy.microphoneUnsupported
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

  if (name === 'NotSupportedError') {
    return new Error(unsupportedMicrophone(copy))
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
  const pcmTapRef = useRef<ScriptProcessorNode | null>(null)
  const stopMeterRef = useRef<(() => void) | null>(null)
  const startedAtRef = useRef(0)
  const heardSpeechRef = useRef(false)
  const meterFailedRef = useRef(false)
  const silenceTriggeredRef = useRef(false)
  const silenceStartedAtRef = useRef<number | null>(null)
  const stopResolverRef = useRef<((recording: MicRecording | null) => void) | null>(null)

  const cleanup = () => {
    stopMeterRef.current?.()
    stopMeterRef.current = null

    pcmTapRef.current?.disconnect()
    pcmTapRef.current = null
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

  const startMeter = (stream: MediaStream, options: MicRecorderOptions) => {
    const audioWindow = window as Window & { webkitAudioContext?: BrowserAudioContext }
    const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

    if (!AudioContextCtor) {
      return
    }

    const recorder = recorderRef.current

    const failMeter = () => {
      if (meterFailedRef.current || recorderRef.current !== recorder || !recorder) {
        return
      }

      meterFailedRef.current = true

      stopMeterRef.current?.()
      stopMeterRef.current = null

      setLevel(0)
      // Deferred: a meter that fails while start() is still running must not
      // re-enter the caller before start() has resolved.
      window.setTimeout(() => {
        if (recorderRef.current === recorder) {
          options.onMeterFailure?.()
        }
      }, 0)
    }

    try {
      const audioContext = new AudioContextCtor()
      const meterStartedAt = audioContext.currentTime * 1_000
      const source = audioContext.createMediaStreamSource(stream)

      audioContextRef.current = audioContext

      if (options.onPcm) {
        startPcmTap(audioContext, source, options)
      }

      // Device and renderer failures must not leave continuous voice waiting
      // forever for a silence notification.
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

      if (audioContext.state === 'suspended') {
        audioContext.resume().catch(failIfCurrent)
      }

      stopMeterRef.current = startVoiceLevelMeter(
        audioContext,
        source,
        (normalized, now, isCurrent) => {
          if (audioContextRef.current !== audioContext) {
            return
          }

          if (document.visibilityState === 'visible') {
            setLevel(normalized)
          }

          options.onLevel?.(normalized)

          if (audioContextRef.current !== audioContext) {
            return
          }

          const speechThreshold = options.silenceLevel ?? 0

          if (speechThreshold > 0 && options.onSilence && !silenceTriggeredRef.current) {
            if (normalized >= speechThreshold) {
              heardSpeechRef.current = true
              silenceStartedAtRef.current = null

              return
            }

            silenceStartedAtRef.current ??= now
            const silenceMs = (heardSpeechRef.current ? options.silenceMs : options.idleSilenceMs) ?? 0
            const quietSince = heardSpeechRef.current ? silenceStartedAtRef.current : meterStartedAt

            if (isCurrent && silenceMs > 0 && now - quietSince >= silenceMs) {
              silenceTriggeredRef.current = true
              stopMeterRef.current?.()
              stopMeterRef.current = null
              options.onSilence()
            }
          }
        },
        failIfCurrent
      )
    } catch {
      failMeter()
    }
  }

  // ScriptProcessor: deprecated but universal, and it needs no worklet module URL in the Electron
  // renderer. The host resamples, so chunks go out at the context's own rate.
  const startPcmTap = (audioContext: AudioContext, source: MediaStreamAudioSourceNode, options: MicRecorderOptions) => {
    const tap = audioContext.createScriptProcessor(4096, 1, 1)
    pcmTapRef.current = tap
    options.onPcmRate?.(audioContext.sampleRate)

    tap.onaudioprocess = event => {
      const input = event.inputBuffer.getChannelData(0)
      const pcm = new Int16Array(input.length)

      for (let i = 0; i < input.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, input[i]))
        pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
      }

      options.onPcm?.(pcm.buffer)
    }

    // A tap only fires while connected through to the destination; its output buffer stays silent.
    source.connect(tap)
    tap.connect(audioContext.destination)
  }

  const start: MicRecorderHandle['start'] = async (options = {}) => {
    if (recorderRef.current) {
      return
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      throw new Error(unsupportedMicrophone(copy))
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
    silenceTriggeredRef.current = false
    silenceStartedAtRef.current = null
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
        meterFailed
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
    startMeter(stream, options)
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

    if (recorder) {
      recorder.ondataavailable = null
      recorder.onerror = null
      recorder.onstop = null

      if (recorder.state !== 'inactive') {
        recorder.stop()
      }
    }

    cleanup()
    resolver?.(null)
  }

  const handle: MicRecorderHandle = { start, stop, cancel }

  return { handle, level, recording }
}
