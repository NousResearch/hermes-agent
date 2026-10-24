/**
 * Client-side mic capture for remote wake word.
 *
 * When the backend arms with `capture: "client"`, PortAudio runs on a headless
 * VM with no mic. The desktop opens getUserMedia here, resamples to 16 kHz
 * mono int16 frames, and pushes them via `wake.feed` so openWakeWord still
 * runs server-side without requiring a server sound device.
 */

const TARGET_RATE = 16_000
const DEFAULT_FRAME = 1280 // 80 ms @ 16 kHz — matches tools/wake_word.py

// Health-monitor tuning for the continuous capture chain. A platform capture
// failure (macOS PLAS IPC delegate error → StopSourceOnError, #119089) leaves
// getUserMedia resolved but the PCM dead: ended track, halted callbacks, or
// endless digital zeros. Without a watchdog the ear shows "listening" forever
// and the gateway detector can never fire.
/** -60 dBFS: live mic noise floors sit well above this; exact zeros do not. */
const SILENCE_PEAK = 0.001
/** ~8 s of digital zeros at 80 ms/frame before the ear is declared deaf. */
const DEFAULT_SILENCE_FRAMES = 100
const DEFAULT_STALL_TIMEOUT_MS = 3000
const DEFAULT_MAX_FEED_FAILURES = 5

export type WakeFeedRequester = (method: string, params?: Record<string, unknown>) => Promise<unknown>

export interface ClientWakeCaptureOptions {
  /** Samples per frame at 16 kHz (from wake.start response). */
  frameLength?: number
  request: WakeFeedRequester
  /**
   * Fatal capture-chain failures only (dead track, stalled graph, sustained
   * silence, refused feeds). Isolated feed RPC blips are retried silently —
   * reporting every one would flap the ear on ordinary network jitter.
   */
  onError?: (error: Error) => void
  /** Consecutive near-silent 16 kHz frames before the ear is declared deaf. */
  silenceFramesThreshold?: number
  /** ms without an onaudioprocess callback before the graph is declared stalled. */
  stallTimeoutMs?: number
  /** Consecutive rejected/failed wake.feed calls before escalating. */
  maxConsecutiveFeedFailures?: number
}

export interface ClientWakeCaptureHandle {
  stop: () => void
  readonly active: boolean
}

function downsampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_RATE) {
    return input
  }

  if (inputRate <= 0) {
    return new Float32Array(0)
  }

  const ratio = inputRate / TARGET_RATE
  const outLen = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLen)

  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio)
    const end = Math.min(input.length, Math.floor((i + 1) * ratio))
    let sum = 0
    let count = 0

    for (let j = start; j < end; j++) {
      sum += input[j] ?? 0
      count++
    }

    out[i] = count > 0 ? sum / count : 0
  }

  return out
}

function floatToInt16LE(input: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(input.length * 2)
  const view = new DataView(buf)

  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i] ?? 0))
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }

  return buf
}

function bytesToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  let binary = ''
  const chunk = 0x8000

  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }

  return btoa(binary)
}

/**
 * Start streaming the default microphone to `wake.feed`.
 * Returns a handle whose `stop()` ends tracks + audio graph.
 */
export async function startClientWakeCapture(options: ClientWakeCaptureOptions): Promise<ClientWakeCaptureHandle> {
  const frameLength = Math.max(160, Math.trunc(options.frameLength || DEFAULT_FRAME))
  const audioWindow = window as Window & { webkitAudioContext?: typeof AudioContext }
  const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

  if (!AudioContextCtor) {
    throw new Error('AudioContext unavailable for client wake capture')
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia unavailable for client wake capture')
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    },
    video: false
  })

  // getUserMedia can resolve with a stillborn stream (no audio track, or an
  // already-ended one after a platform capture error). Starting the graph on
  // that feeds the detector silence forever — fail loudly instead (#119089).
  const audioTracks = stream.getAudioTracks()

  if (audioTracks.length === 0 || audioTracks.every(track => track.readyState === 'ended')) {
    stream.getTracks().forEach(track => track.stop())

    throw new Error('microphone track unavailable for client wake capture')
  }

  const silenceFramesThreshold = Math.max(1, Math.trunc(options.silenceFramesThreshold ?? DEFAULT_SILENCE_FRAMES))
  const stallTimeoutMs = Math.max(250, options.stallTimeoutMs ?? DEFAULT_STALL_TIMEOUT_MS)
  const maxFeedFailures = Math.max(1, Math.trunc(options.maxConsecutiveFeedFailures ?? DEFAULT_MAX_FEED_FAILURES))

  const context = new AudioContextCtor()
  const source = context.createMediaStreamSource(stream)
  // ScriptProcessor is deprecated but widely available and simple for PCM export.
  // Buffer size 4096 keeps callback rate reasonable on desktop.
  const processor = context.createScriptProcessor(4096, 1, 1)
  const mute = context.createGain()
  mute.gain.value = 0

  let pending = new Float32Array(0)
  let stopped = false
  let failed = false
  let silentFrames = 0
  let feedFailures = 0
  let stallTimer: ReturnType<typeof setTimeout> | undefined

  const handle: ClientWakeCaptureHandle = {
    get active() {
      return !stopped
    },
    stop() {
      if (stopped) {
        return
      }

      stopped = true
      queue.length = 0

      if (stallTimer !== undefined) {
        clearTimeout(stallTimer)
        stallTimer = undefined
      }

      try {
        processor.disconnect()
        source.disconnect()
        mute.disconnect()
      } catch {
        // ignore
      }

      void context.close().catch(() => undefined)
      stream.getTracks().forEach(t => t.stop())
    }
  }

  // Fatal capture-chain failure: tear the graph down exactly once and report.
  // stop() is idempotent, so late track-ended events after a manual stop stay silent.
  const fail = (error: Error) => {
    if (failed || stopped) {
      return
    }

    failed = true
    handle.stop()
    options.onError?.(error)
  }

  const armStallTimer = () => {
    if (stopped || failed) {
      return
    }

    if (stallTimer !== undefined) {
      clearTimeout(stallTimer)
    }

    stallTimer = setTimeout(() => {
      fail(new Error('client wake capture stalled: no microphone audio callbacks — the OS capture chain may have died'))
    }, stallTimeoutMs)
  }

  // A platform capture error (macOS StopSourceOnError) ends the track after a
  // successful getUserMedia. Without this the graph feeds zeros forever.
  for (const track of audioTracks) {
    track.onended = () => {
      fail(new Error('microphone track ended during client wake capture — the OS capture chain may have died'))
    }
  }

  // Bounded ordered queue of 16 kHz frames. We never drop the frame that is
  // currently being sent; under remote latency we drop the oldest queued
  // frames so the detector still sees contiguous recent PCM rather than gaps
  // from fire-and-forget discard-while-inflight.
  const MAX_QUEUED_FRAMES = 24 // ~1.9s at 80 ms/frame
  // Coalesce queued frames into one wake.feed call (backend splits them back
  // into engine frames). 4 × 80 ms ≈ 3 RPCs/s steady-state instead of 12.5.
  const MAX_FRAMES_PER_FEED = 4
  const queue: Float32Array[] = []
  let draining = false

  const noteFeedResult = (accepted: boolean, reason: string | null) => {
    if (stopped || failed) {
      return
    }

    if (accepted) {
      feedFailures = 0

      return
    }

    // The server dropped our PCM (lease lost, wrong owner, detector gone).
    // One blip is ordinary network jitter; a run of them means the ear is
    // armed but permanently deaf — escalate instead of staying "listening".
    feedFailures += 1

    if (feedFailures >= maxFeedFailures) {
      fail(
        new Error(
          `client wake capture deaf: wake.feed refused ${feedFailures} consecutive frames` +
            (reason ? ` (${reason})` : '') +
            ' — re-toggle the ear to re-arm'
        )
      )
    }
  }

  const drainQueue = async () => {
    if (draining) {
      return
    }

    draining = true

    try {
      while (!stopped && !failed && queue.length > 0) {
        const batch = queue.splice(0, MAX_FRAMES_PER_FEED)

        if (batch.length === 0) {
          break
        }

        try {
          const merged = new Float32Array(batch.length * frameLength)
          batch.forEach((frame, i) => merged.set(frame, i * frameLength))
          const pcm = floatToInt16LE(merged)

          const result = (await options.request('wake.feed', {
            pcm: bytesToBase64(pcm),
            sample_rate: TARGET_RATE
          })) as { fed?: boolean; reason?: string | null } | null | undefined

          // Absent on older backends that predate the {fed} envelope — only an
          // explicit fed:false counts as a refusal. Thrown RPC errors land below.
          noteFeedResult(!result || result.fed !== false, result?.reason ?? null)
        } catch {
          // Keep draining later frames; one failed RPC should not freeze the ear.
          noteFeedResult(false, null)
        }
      }
    } finally {
      draining = false

      if (!stopped && !failed && queue.length > 0) {
        void drainQueue()
      }
    }
  }

  const enqueueFrame = (frame: Float32Array) => {
    if (stopped) {
      return
    }

    queue.push(frame)

    while (queue.length > MAX_QUEUED_FRAMES) {
      queue.shift()
    }

    void drainQueue()
  }

  processor.onaudioprocess = event => {
    if (stopped || failed) {
      return
    }

    armStallTimer()

    const input = event.inputBuffer.getChannelData(0)
    const at16k = downsampleTo16k(input, context.sampleRate)
    // Append to pending and emit full frames
    const merged = new Float32Array(pending.length + at16k.length)
    merged.set(pending, 0)
    merged.set(at16k, pending.length)
    let offset = 0

    while (offset + frameLength <= merged.length) {
      const frame = merged.subarray(offset, offset + frameLength)
      offset += frameLength

      // A dead capture chain delivers endless digital zeros. Live mics never
      // do — their noise floor sits far above SILENCE_PEAK — so a long run of
      // zeros means the detector is being fed silence, not speech (#119089).
      let peak = 0

      for (let i = 0; i < frame.length; i++) {
        const abs = Math.abs(frame[i] ?? 0)

        if (abs > peak) {
          peak = abs
        }
      }

      if (peak < SILENCE_PEAK) {
        silentFrames += 1

        if (silentFrames >= silenceFramesThreshold) {
          fail(
            new Error(
              'client wake capture hears only silence — the microphone delivers no audio; check OS mic access and re-toggle the ear'
            )
          )

          return
        }

        continue
      }

      silentFrames = 0
      enqueueFrame(new Float32Array(frame))
    }

    pending = merged.subarray(offset)
  }

  source.connect(processor)
  processor.connect(mute)
  mute.connect(context.destination)

  if (context.state === 'suspended') {
    await context.resume().catch(() => undefined)
  }

  armStallTimer()

  return handle
}
