import { useEffect, useRef, useState } from 'react'

import { float32ToInt16Pcm, resampleTo16k } from '@/lib/pcm'

type BrowserAudioContext = typeof AudioContext

export interface MicRecorderOptions {
  onLevel?: (level: number) => void
  onError?: (error: Error) => void
  onSilence?: () => void
  silenceLevel?: number
  silenceMs?: number
  idleSilenceMs?: number
  /** Live PCM tap: receives 16 kHz mono s16le chunks as they are captured
   *  (used by streaming dictation — the chunks feed the host's live STT
   *  relay while the user is still speaking). Only delivered when a 16 kHz
   *  AudioContext is available; the MediaRecorder blob path is unaffected. */
  onPcm?: (chunk: ArrayBuffer) => void
}

export interface MicRecording {
  audio: Blob
  durationMs: number
  heardSpeech: boolean
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

function micError(error: unknown, copy: MicRecorderErrorCopy): Error {
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
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const animationRef = useRef<number | null>(null)
  const startedAtRef = useRef(0)
  const heardSpeechRef = useRef(false)
  const silenceTriggeredRef = useRef(false)
  const silenceStartedAtRef = useRef<number | null>(null)
  const stopResolverRef = useRef<((recording: MicRecording | null) => void) | null>(null)

  const cleanup = () => {
    if (animationRef.current) {
      window.cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }

    void audioContextRef.current?.close()
    audioContextRef.current = null
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

    try {
      const wantsPcm = typeof options.onPcm === 'function'
      const audioContext = new AudioContextCtor()
      const analyser = audioContext.createAnalyser()
      const source = audioContext.createMediaStreamSource(stream)

      analyser.fftSize = 256
      const data = new Uint8Array(analyser.fftSize)

      source.connect(analyser)
      audioContextRef.current = audioContext

      if (wantsPcm) {
        // The context runs at the device rate — macOS can clamp a requested
        // 16 kHz context to the hardware rate, and a silent rate mismatch
        // used to starve the streaming leg. The tap resamples in JS, so
        // onPcm ALWAYS delivers 16 kHz mono s16le regardless of context rate.
        // ScriptProcessor is deprecated but universally supported and needs
        // no worklet module URL in the Electron renderer.
        //
        // Two input channels + explicit L/R average: a mono-configured node
        // relies on the browser's downmix, which has produced silent input in
        // some Electron/macOS combos — averaging ourselves is deterministic
        // whether the track is mono or stereo.
        const processor = audioContext.createScriptProcessor(4096, 2, 1)
        processorRef.current = processor
        const contextRate = audioContext.sampleRate

        // The tap only hears what is connected to it: source → processor,
        // or onaudioprocess fires with SILENT buffers (201 chunks of pure
        // zero — caught by the server's energy probe, avg_abs_level=0).
        source.connect(processor)

        processor.onaudioprocess = event => {
          const left = event.inputBuffer.getChannelData(0)
          const right = event.inputBuffer.numberOfChannels > 1 ? event.inputBuffer.getChannelData(1) : null
          const input = new Float32Array(left.length)

          for (let i = 0; i < left.length; i += 1) {
            input[i] = right ? (left[i] + right[i]) * 0.5 : left[i]
          }

          const resampled = resampleTo16k(input, contextRate)
          options.onPcm?.(float32ToInt16Pcm(resampled))
        }

        processor.connect(audioContext.destination)
      }

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
          if (normalized >= speechThreshold) {
            heardSpeechRef.current = true
            silenceStartedAtRef.current = null
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
    } catch {
      setLevel(0)
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
        heardSpeech
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
