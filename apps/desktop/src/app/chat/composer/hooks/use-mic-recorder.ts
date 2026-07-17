import { useCallback, useEffect, useRef, useState } from 'react'

type BrowserAudioContext = typeof AudioContext

export interface MicRecorderOptions {
  onLevel?: (level: number) => void
  onError?: (error: Error) => void
  onSilence?: () => void
  silenceLevel?: number
  silenceMs?: number
  idleSilenceMs?: number
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
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const animationRef = useRef<number | null>(null)
  const startedAtRef = useRef(0)
  const heardSpeechRef = useRef(false)
  const silenceTriggeredRef = useRef(false)
  const silenceStartedAtRef = useRef<number | null>(null)
  const stopResolverRef = useRef<((recording: MicRecording | null) => void) | null>(null)

  const stopMeter = useCallback(() => {
    if (animationRef.current) {
      window.cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    setLevel(0)
  }, [])

  // Tear down the per-turn audio graph (meter loop, analyser, AudioContext).
  // The AudioContext is intentionally NOT kept alive between turns: the browser
  // can suspend an idle context, after which the analyser reads silence and
  // speech detection silently stops (the turn then idles out after ~12s). It's
  // cheap to recreate, so we rebuild it fresh on every re-arm.
  const teardownAudioGraph = useCallback(() => {
    stopMeter()
    sourceRef.current?.disconnect()
    sourceRef.current = null
    analyserRef.current = null
    void audioContextRef.current?.close()
    audioContextRef.current = null
  }, [stopMeter])

  // Soft stop between turns: drop the per-turn MediaRecorder + audio graph but
  // KEEP the mic STREAM alive, so the next turn re-arms without a fresh
  // getUserMedia (that round-trip was the multi-second post-TTS mic delay).
  const softCleanup = useCallback(() => {
    teardownAudioGraph()
    recorderRef.current = null
    setRecording(false)
    silenceTriggeredRef.current = false
  }, [teardownAudioGraph])

  // Full release: also stop the mic tracks. Used when the conversation
  // ends/cancels, on a recorder error, or on unmount.
  const releaseStream = useCallback(() => {
    softCleanup()
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
  }, [softCleanup])

  useEffect(() => () => releaseStream(), [releaseStream])

  const startMeter = (stream: MediaStream, options: MicRecorderOptions) => {
    const audioWindow = window as Window & { webkitAudioContext?: BrowserAudioContext }
    const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

    if (!AudioContextCtor) {
      return
    }

    try {
      // Fresh AudioContext per turn (see teardownAudioGraph) — built on the
      // kept-alive stream, so this is cheap and always starts in 'running'.
      const audioContext = new AudioContextCtor()
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256

      const source = audioContext.createMediaStreamSource(stream)
      source.connect(analyser)

      audioContextRef.current = audioContext
      analyserRef.current = analyser
      sourceRef.current = source

      const data = new Uint8Array(analyser.fftSize)

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

    // Reuse a still-live stream from a previous turn so we skip the costly
    // requestMicrophoneAccess + getUserMedia round-trip — that round-trip was
    // the seconds-long re-arm delay after the assistant finished speaking.
    let stream = streamRef.current

    if (!stream || !stream.getTracks().some(track => track.readyState === 'live')) {
      const permitted = await window.hermesDesktop?.requestMicrophoneAccess?.()

      if (permitted === false) {
        throw new Error(copy.microphoneAccessDenied)
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true }
        })
      } catch (error) {
        throw micError(error, copy)
      }

      // A brand-new stream invalidates any analyser graph wired to the old one.
      sourceRef.current?.disconnect()
      sourceRef.current = null
      analyserRef.current = null
      void audioContextRef.current?.close()
      audioContextRef.current = null
      streamRef.current = stream
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
      softCleanup()

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
      releaseStream()
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
        softCleanup()
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

    releaseStream()
    resolver?.(null)
  }

  const handle: MicRecorderHandle = { start, stop, cancel }

  return { handle, level, recording }
}
