// VAD barge-in: watch the mic while TTS plays, fire the moment the user talks
// over it, and CAPTURE what they say. Detection alone loses the first words —
// by the time sustained speech trips the trigger and a fresh recorder spins
// up, "stop, actually—" has become "actually—". So a MediaRecorder runs on
// the monitor's stream the whole time (pre-roll), and once tripped it keeps
// rolling until the user goes quiet, delivering the complete utterance.
//
// Echo cancellation strips the app's own speaker output from the capture, the
// noise floor is calibrated while playback is already audible, and the
// sustained window filters coughs/thumps — mirrors
// tools/voice_mode.listen_for_speech on the Python surfaces.

const CALIBRATION_MS = 400
const SUSTAINED_MS = 300
const MIN_TRIGGER_LEVEL = 0.075 // matches the voice loop's silenceLevel
const PRE_ROLL_RESTART_MS = 5_000 // cap pre-roll: restart the recorder while quiet
const UTTERANCE_SILENCE_MS = 1_250 // matches the voice loop's silenceMs
const UTTERANCE_MAX_MS = 30_000

export interface BargeMonitorCallbacks {
  /** Sustained speech detected — cut playback now. */
  onSpeech: () => void
  /**
   * The interrupting utterance, complete from its first syllable (pre-roll
   * included), delivered once the user goes quiet. `null` when capture was
   * unavailable — fall back to normal listening.
   */
  onUtterance?: (audio: Blob | null) => void
}

export function monitorSpeechDuringPlayback(callbacks: BargeMonitorCallbacks): () => void {
  let disposed = false
  let stream: MediaStream | null = null
  let context: AudioContext | null = null
  let frame: number | null = null
  let recorder: MediaRecorder | null = null
  let chunks: Blob[] = []
  let mimeType = ''

  const cleanup = () => {
    disposed = true

    if (frame !== null) {
      window.cancelAnimationFrame(frame)
      frame = null
    }

    if (recorder && recorder.state !== 'inactive') {
      recorder.ondataavailable = null
      recorder.onstop = null

      try {
        recorder.stop()
      } catch {
        // already stopped
      }
    }

    recorder = null
    chunks = []
    void context?.close().catch(() => undefined)
    context = null
    stream?.getTracks().forEach(track => track.stop())
    stream = null
  }

  const startSegment = () => {
    if (!stream || typeof MediaRecorder === 'undefined') {
      return
    }

    mimeType =
      ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'].find(type =>
        MediaRecorder.isTypeSupported(type)
      ) ?? ''

    try {
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    } catch {
      recorder = null

      return
    }

    chunks = []

    recorder.ondataavailable = event => {
      if (event.data.size > 0) {
        chunks.push(event.data)
      }
    }

    recorder.start(250)
  }

  /** Restart the recorder to drop stale pre-roll — only valid while quiet. */
  const rotateSegment = () => {
    if (!recorder || recorder.state === 'inactive') {
      return
    }

    recorder.ondataavailable = null
    recorder.onstop = null

    try {
      recorder.stop()
    } catch {
      // already stopped
    }

    startSegment()
  }

  const finishCapture = () => {
    const active = recorder
    const type = active?.mimeType || mimeType || 'audio/webm'

    if (!active || active.state === 'inactive') {
      cleanup()
      callbacks.onUtterance?.(chunks.length ? new Blob(chunks, { type }) : null)

      return
    }

    active.onstop = () => {
      const audio = chunks.length ? new Blob(chunks, { type }) : null

      cleanup()
      callbacks.onUtterance?.(audio)
    }

    active.stop()
  }
  void (async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      })

      if (disposed) {
        cleanup()

        return
      }

      startSegment()

      context = new AudioContext()
      const analyser = context.createAnalyser()
      analyser.fftSize = 256
      context.createMediaStreamSource(stream).connect(analyser)

      const data = new Uint8Array(analyser.fftSize)
      const startedAt = Date.now()
      const floorSamples: number[] = []
      let segmentStartedAt = Date.now()
      let speechStartedAt: number | null = null
      let tripped = false
      let trippedAt = 0
      let quietSince: number | null = null

      const tick = () => {
        if (disposed) {
          return
        }

        analyser.getByteTimeDomainData(data)

        let sum = 0

        for (const value of data) {
          const centered = value - 128
          sum += centered * centered
        }

        const level = Math.min(1, Math.sqrt(sum / data.length) / 42)
        const now = Date.now()

        if (!tripped && now - startedAt < CALIBRATION_MS) {
          floorSamples.push(level)
        } else if (!tripped) {
          const floor = floorSamples.length ? [...floorSamples].sort((a, b) => a - b)[floorSamples.length >> 1] : 0
          const trigger = Math.max(MIN_TRIGGER_LEVEL, floor * 3.5)

          if (level >= trigger) {
            speechStartedAt ??= now

            if (now - speechStartedAt >= SUSTAINED_MS) {
              tripped = true
              trippedAt = now
              quietSince = null
              callbacks.onSpeech()

              if (!callbacks.onUtterance || !recorder) {
                cleanup()
                callbacks.onUtterance?.(null)

                return
              }
            }
          } else {
            speechStartedAt = null

            // Bound the pre-roll while quiet so the utterance blob doesn't
            // accumulate the whole playback (rotating mid-speech would lose
            // the onset — the whole point).
            if (now - segmentStartedAt >= PRE_ROLL_RESTART_MS) {
              rotateSegment()
              segmentStartedAt = now
            }
          }
        } else {
          // Tripped: keep recording until the user goes quiet (endpoint).
          // Playback is already stopped, so plain silence-vs-speech works.
          if (level >= MIN_TRIGGER_LEVEL) {
            quietSince = null
          } else {
            quietSince ??= now
          }

          if ((quietSince && now - quietSince >= UTTERANCE_SILENCE_MS) || now - trippedAt >= UTTERANCE_MAX_MS) {
            finishCapture()

            return
          }
        }

        frame = window.requestAnimationFrame(tick)
      }

      tick()
    } catch {
      cleanup()
    }
  })()

  return cleanup
}
