// Full-duplex VAD monitor: watch the mic across the agent turn — while the
// model is generating (no audio yet) AND while TTS plays — fire the moment the
// user talks over either phase, and CAPTURE what they say. Detection alone
// loses the first words — by the time sustained speech trips the trigger and a
// fresh recorder spins up, "stop, actually—" has become "actually—". So a
// MediaRecorder runs on the monitor's stream the whole time (pre-roll), and
// once tripped it keeps rolling until the user goes quiet, delivering the
// complete utterance.
//
// Phase-aware trigger (mirrors tools/voice_mode.full_duplex_listen on the
// Python surfaces):
// - The noise floor is calibrated from QUIET samples only — while no TTS audio
//   is flowing — and HELD through playback. Calibrating while the speaker is
//   audible bakes bleed into the floor and makes the trigger unreachable
//   (echoCancellation does not reliably cancel same-app playback on Windows).
// - During playback the trigger is additionally clamped up to a minimum so
//   bleed alone can't trip it, and capped so speech always remains reachable.
// - A short grace window after playback onset suppresses the start transient.
// - `voice.barge_in_threshold_multiplier` (the backend's sensitivity knob)
//   scales the quiet-floor multiplier AND the playback clamp relative to the
//   backend default, so a Bluetooth HFP headset whose mic sits below the stock
//   playback clamp can still interrupt (#117801). Unset keeps the stock levels.
// - Detection is a windowed majority (>=80% of the last SUSTAINED_MS above
//   trigger) so intra-word energy dips don't reset progress.

import { closeMeterContext, meterContextsClosed } from '@/lib/mic-meter-context'
import { startVoiceLevelMeter } from '@/lib/voice-level-meter'
import { $voiceSilenceMs } from '@/store/voice-prefs'

const CALIBRATION_MS = 400
const SUSTAINED_MS = 300
const SUSTAINED_MAJORITY = 0.8
const MIN_TRIGGER_LEVEL = 0.075 // matches the voice loop's silenceLevel
const FLOOR_MULTIPLIER = 3.5
// Playback clamps, scaled from the Python constants (int16 RMS 1500 / 4000
// ≈ byte-domain level 0.14 / 0.37 with the /42 normalization below).
const PLAYBACK_MIN_TRIGGER_LEVEL = 0.14
const TRIGGER_CEILING_LEVEL = 0.37
// Backend default for voice.barge_in_threshold_multiplier
// (tools/voice_mode.DEFAULT_BARGE_MULTIPLIER); the levels above are tuned for it.
export const DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER = 3
const PLAYBACK_GRACE_MS = 500
const PLAYBACK_GAP_FOR_GRACE_MS = 1_000
const FLOOR_SAMPLE_CAP = 200 // ~3s of quiet-phase levels at rAF cadence
const PRE_ROLL_RESTART_MS = 5_000 // cap pre-roll: restart the recorder while quiet
// The utterance endpoint shares the voice loop's `silenceMs` ($voiceSilenceMs,
// seeded from `voice.silence_duration`) so barge-in capture ends on the same
// silence window the loop uses; read live per frame so a config refresh
// mid-turn keeps the two matched.
const UTTERANCE_MAX_MS = 30_000

export interface BargeMonitorCallbacks {
  /** Sustained speech detected — cut playback / interrupt the turn now. */
  onSpeech: () => void
  /**
   * The interrupting utterance, complete from its first syllable (pre-roll
   * included), delivered once the user goes quiet. `null` when capture was
   * unavailable — fall back to normal listening.
   */
  onUtterance?: (audio: Blob | null) => void
  /**
   * Is TTS audio flowing RIGHT NOW? Drives the phase-aware trigger. Omitted
   * (legacy playback-only callers) means "always playing", which preserves
   * the old behavior of a monitor opened at playback start.
   */
  isPlaying?: () => boolean
  /**
   * `voice.barge_in_threshold_multiplier`. Lower = more sensitive. Omitted,
   * null, or non-positive keeps the stock trigger levels.
   */
  thresholdMultiplier?: number | null
}

export interface BargeTriggerLevels {
  /** Trigger = quiet floor x this (never below MIN_TRIGGER_LEVEL). */
  floorMultiplier: number
  /** Minimum trigger while TTS plays, so speaker bleed alone can't trip. */
  playbackMinTrigger: number
}

/** Trigger levels for a configured `voice.barge_in_threshold_multiplier`,
 *  scaled from the stock levels by its ratio to the backend default. */
export function bargeInTriggerLevels(thresholdMultiplier?: number | null): BargeTriggerLevels {
  const scale =
    typeof thresholdMultiplier === 'number' && Number.isFinite(thresholdMultiplier) && thresholdMultiplier > 0
      ? thresholdMultiplier / DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER
      : 1

  return { floorMultiplier: FLOOR_MULTIPLIER * scale, playbackMinTrigger: PLAYBACK_MIN_TRIGGER_LEVEL * scale }
}

export function monitorSpeechDuringPlayback(callbacks: BargeMonitorCallbacks): () => void {
  const { floorMultiplier, playbackMinTrigger } = bargeInTriggerLevels(callbacks.thresholdMultiplier)
  let disposed = false
  let stream: MediaStream | null = null
  let context: AudioContext | null = null
  let stopMeter: (() => void) | null = null
  let recorder: MediaRecorder | null = null
  let chunks: Blob[] = []
  let mimeType = ''

  const cleanup = () => {
    disposed = true

    stopMeter?.()
    stopMeter = null

    if (recorder) {
      recorder.ondataavailable = null
      recorder.onstop = null

      if (recorder.state !== 'inactive') {
        try {
          recorder.stop()
        } catch {
          // already stopped
        }
      }
    }

    recorder = null
    chunks = []
    closeMeterContext(context)
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
    stopMeter?.()
    stopMeter = null
    const active = recorder
    const type = active?.mimeType || mimeType || 'audio/webm'

    if (!active || active.state === 'inactive') {
      const audio = chunks.length ? new Blob(chunks, { type }) : null

      cleanup()
      callbacks.onUtterance?.(audio)

      return
    }

    active.onstop = () => {
      const audio = chunks.length ? new Blob(chunks, { type }) : null

      cleanup()
      callbacks.onUtterance?.(audio)
    }

    active.stop()
  }

  // No mediaDevices (an insecure origin): there is no mic to monitor. The conversation's own
  // listening start already reports that, through the recorder's "microphone unsupported" copy.
  if (!navigator.mediaDevices?.getUserMedia) {
    return cleanup
  }
  void (async () => {
    try {
      // Don't open a second capture context while the recorder's is still
      // closing (#75329).
      await meterContextsClosed()

      if (disposed) {
        return
      }

      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      })

      if (disposed) {
        cleanup()

        return
      }

      startSegment()

      context = new AudioContext()
      const source = context.createMediaStreamSource(stream)
      const floorSamples: number[] = []
      const recentAbove: { above: boolean; at: number }[] = []
      let calibratedSince: number | null = null
      let floorLocked = false
      let quietFloor = 0
      let segmentStartedAt = context.currentTime * 1_000
      let wasPlaying = false
      let playbackSeen = false
      let lastPlayingAt = 0
      let graceUntil = 0
      let tripped = false
      let trippedAt = 0
      let quietSince: number | null = null

      const failMeter = () => {
        if (disposed) {
          return
        }

        cleanup()

        if (tripped) {
          // Release the conversation's pending capture without transcribing a
          // potentially truncated command. Explicit cancellation stays silent.
          callbacks.onUtterance?.(null)
        }
      }

      const pushFloorSample = (level: number) => {
        floorSamples.push(level)

        if (floorSamples.length > FLOOR_SAMPLE_CAP) {
          floorSamples.shift()
        }

        quietFloor = [...floorSamples].sort((a, b) => a - b)[floorSamples.length >> 1] ?? 0
      }

      const updateTrigger = (level: number, now: number, playing: boolean) => {
        // Calibrate from quiet samples, then hold the floor through playback.
        if (!floorLocked) {
          if (!playing) {
            calibratedSince ??= now
            pushFloorSample(level)
          }

          if (playing || (calibratedSince !== null && now - calibratedSince >= CALIBRATION_MS)) {
            floorLocked = true
          }
        }

        // Sentence gaps must not chain playback-onset grace windows.
        if (playing && !wasPlaying) {
          if (!playbackSeen || now - lastPlayingAt >= PLAYBACK_GAP_FOR_GRACE_MS) {
            graceUntil = now + PLAYBACK_GRACE_MS
          }

          playbackSeen = true
        }

        wasPlaying = playing

        if (playing) {
          lastPlayingAt = now
        }

        let trigger = Math.max(MIN_TRIGGER_LEVEL, quietFloor * floorMultiplier)

        if (playing) {
          trigger = Math.min(Math.max(trigger, playbackMinTrigger), TRIGGER_CEILING_LEVEL)
        }

        // Track ambient drift while quiet and below trigger.
        if (floorLocked && !playing && level < trigger) {
          pushFloorSample(level)
        }

        return trigger
      }

      const captureLevel = (level: number, now: number, isCurrent: boolean) => {
        // Playback/generation was already cut; silence ends the captured turn.
        if (level >= MIN_TRIGGER_LEVEL) {
          quietSince = null
        } else {
          quietSince ??= now
        }

        if (
          isCurrent &&
          ((quietSince !== null && now - quietSince >= $voiceSilenceMs.get()) || now - trippedAt >= UTTERANCE_MAX_MS)
        ) {
          finishCapture()
        }
      }

      const onLevel = (level: number, now: number, isCurrent: boolean) => {
        if (disposed) {
          return
        }

        if (tripped) {
          captureLevel(level, now, isCurrent)

          return
        }

        const playing = callbacks.isPlaying ? callbacks.isPlaying() : true
        const trigger = updateTrigger(level, now, playing)
        const above = floorLocked && level >= trigger && now >= graceUntil
        recentAbove.push({ above, at: now })

        while (recentAbove.length && now - recentAbove[0].at > SUSTAINED_MS) {
          recentAbove.shift()
        }

        const aboveCount = recentAbove.reduce((count, sample) => count + (sample.above ? 1 : 0), 0)
        const spanMs = recentAbove.length ? now - recentAbove[0].at : 0

        if (
          above &&
          spanMs >= SUSTAINED_MS * SUSTAINED_MAJORITY &&
          aboveCount >= recentAbove.length * SUSTAINED_MAJORITY
        ) {
          tripped = true
          trippedAt = now
          quietSince = null
          callbacks.onSpeech()

          if (disposed) {
            return
          }

          if (!callbacks.onUtterance || !recorder) {
            cleanup()
            callbacks.onUtterance?.(null)
          }
        } else if (isCurrent && !above && now - segmentStartedAt >= PRE_ROLL_RESTART_MS) {
          // Bound quiet pre-roll only after queued speech has been considered.
          rotateSegment()
          segmentStartedAt = now
        }
      }

      stopMeter = startVoiceLevelMeter(context, source, onLevel, failMeter)

      if (context.state === 'suspended') {
        void context.resume().catch(failMeter)
      }
    } catch {
      cleanup()
    }
  })()

  return cleanup
}
