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

// How long to listen to a candidate input before judging it dead (ms).
const PROBE_MS = 700

// Virtual/loopback inputs enumerate exactly like real microphones but emit
// bit-perfect digital silence unless something is routed into them. macOS boxes
// commonly carry several (BlackHole, MMAudio, Loopback, Soundflower), and
// Chromium will happily hand one back as the "default" device — which is how a
// wake word ends up armed, streaming, and permanently deaf. Deprioritise them.
const VIRTUAL_INPUT_PATTERNS = [
  'blackhole',
  'mmaudio',
  'loopback',
  'soundflower',
  'ui sounds',
  'aggregate',
  'multi-output',
  'virtual',
  'ndi',
  'obs'
]

const isVirtualInput = (label: string): boolean => {
  const l = label.toLowerCase()

  return VIRTUAL_INPUT_PATTERNS.some(p => l.includes(p))
}

export type WakeFeedRequester = (method: string, params?: Record<string, unknown>) => Promise<unknown>

export interface ClientWakeCaptureOptions {
  /** Samples per frame at 16 kHz (from wake.start response). */
  frameLength?: number
  request: WakeFeedRequester
  onError?: (error: Error) => void
  /** Reports which input won and whether it carried any signal at all. */
  onDeviceChosen?: (info: { label: string; live: boolean }) => void
}

export interface ClientWakeCaptureHandle {
  stop: () => void
  readonly active: boolean
  /** Label of the input actually opened (for the ear tooltip / logs). */
  readonly deviceLabel: string
  /** False when every candidate input returned pure digital silence. */
  readonly live: boolean
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

/** Audio constraints for the wake stream.
 *
 * `echoCancellation` matters: Hermes speaks its replies through the same
 * headset it listens on, and without AEC its own TTS can trigger the wake word.
 * `noiseSuppression` is deliberately OFF — an aggressive gate can zero out a
 * quiet room entirely, which both hurts detection and would defeat the
 * liveness probe below. */
const wakeAudioConstraints = (deviceId?: string): MediaTrackConstraints => ({
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: false,
  autoGainControl: true,
  ...(deviceId ? { deviceId: { exact: deviceId } } : {})
})

/**
 * Candidate inputs, best first: real devices before known virtual loopbacks.
 * An empty list means "just take the browser default".
 */
async function orderedInputCandidates(): Promise<Array<{ deviceId: string; label: string }>> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    const inputs = devices
      .filter(d => d.kind === 'audioinput' && d.deviceId && d.deviceId !== 'communications')
      .map(d => ({ deviceId: d.deviceId, label: d.label || d.deviceId }))
    // Stable partition — real inputs keep their enumeration order (which puts
    // the system default first), virtual loopbacks go to the back as fallback.
    return [...inputs.filter(d => !isVirtualInput(d.label)), ...inputs.filter(d => isVirtualInput(d.label))]
  } catch {
    return []
  }
}

/**
 * True when the stream carries any nonzero sample within PROBE_MS.
 *
 * This is an exact-zero test, not a loudness threshold: a real microphone in a
 * silent room still has a noise floor, while a dead or virtual input returns
 * bit-perfect zeros forever. That distinction is the whole point — a threshold
 * would reject a working mic just for being in a quiet room.
 */
async function streamIsLive(context: AudioContext, stream: MediaStream): Promise<boolean> {
  const source = context.createMediaStreamSource(stream)
  const analyser = context.createAnalyser()
  analyser.fftSize = 2048
  source.connect(analyser)

  const buf = new Float32Array(analyser.fftSize)
  const deadline = performance.now() + PROBE_MS

  try {
    while (performance.now() < deadline) {
      analyser.getFloatTimeDomainData(buf)

      for (let i = 0; i < buf.length; i++) {
        if (buf[i] !== 0) {
          return true
        }
      }

      await new Promise(resolve => setTimeout(resolve, 50))
    }

    return false
  } finally {
    try {
      source.disconnect()
      analyser.disconnect()
    } catch {
      // ignore
    }
  }
}

/**
 * Open the first input that actually carries audio.
 *
 * Falls back to the first candidate (still streaming, just silent) so the ear
 * arms rather than failing outright — the backend surfaces the silence and the
 * handle reports `live: false` for the tooltip.
 */
async function openLiveInput(
  context: AudioContext
): Promise<{ stream: MediaStream; label: string; live: boolean }> {
  const candidates = await orderedInputCandidates()
  let fallback: { stream: MediaStream; label: string } | null = null

  for (const candidate of candidates.length ? candidates : [null]) {
    let stream: MediaStream

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: wakeAudioConstraints(candidate?.deviceId),
        video: false
      })
    } catch {
      continue // device vanished / blocked — try the next one
    }

    const label = candidate?.label || stream.getAudioTracks()[0]?.label || 'system default'

    if (await streamIsLive(context, stream)) {
      fallback?.stream.getTracks().forEach(t => t.stop())

      return { stream, label, live: true }
    }

    if (fallback) {
      stream.getTracks().forEach(t => t.stop())
    } else {
      fallback = { stream, label }
    }
  }

  if (fallback) {
    return { ...fallback, live: false }
  }

  throw new Error('No usable microphone for client wake capture')
}

/**
 * Start streaming the microphone to `wake.feed`.
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

  const context = new AudioContextCtor()

  if (context.state === 'suspended') {
    await context.resume().catch(() => undefined)
  }

  let stream: MediaStream
  let deviceLabel: string
  let live: boolean

  try {
    ;({ stream, label: deviceLabel, live } = await openLiveInput(context))
  } catch (error) {
    void context.close().catch(() => undefined)
    throw error
  }

  options.onDeviceChosen?.({ label: deviceLabel, live })

  const source = context.createMediaStreamSource(stream)
  // ScriptProcessor is deprecated but widely available and simple for PCM export.
  // Buffer size 4096 keeps callback rate reasonable on desktop.
  const processor = context.createScriptProcessor(4096, 1, 1)
  const mute = context.createGain()
  mute.gain.value = 0

  let pending = new Float32Array(0)
  let stopped = false
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

  const drainQueue = async () => {
    if (draining) {
      return
    }

    draining = true

    try {
      while (!stopped && queue.length > 0) {
        const batch = queue.splice(0, MAX_FRAMES_PER_FEED)

        if (batch.length === 0) {
          break
        }

        try {
          const merged = new Float32Array(batch.length * frameLength)
          batch.forEach((frame, i) => merged.set(frame, i * frameLength))
          const pcm = floatToInt16LE(merged)
          await options.request('wake.feed', {
            pcm: bytesToBase64(pcm),
            sample_rate: TARGET_RATE
          })
        } catch (error) {
          options.onError?.(error instanceof Error ? error : new Error(String(error)))
          // Keep draining later frames; one failed RPC should not freeze the ear.
        }
      }
    } finally {
      draining = false

      if (!stopped && queue.length > 0) {
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
    if (stopped) {
      return
    }

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

  return {
    get active() {
      return !stopped
    },
    get deviceLabel() {
      return deviceLabel
    },
    get live() {
      return live
    },
    stop() {
      if (stopped) {
        return
      }

      stopped = true
      queue.length = 0

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
}
