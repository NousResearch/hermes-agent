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

export type WakeFeedRequester = (
  method: string,
  params?: Record<string, unknown>
) => Promise<unknown>

export interface ClientWakeCaptureOptions {
  /** Samples per frame at 16 kHz (from wake.start response). */
  frameLength?: number
  request: WakeFeedRequester
  onError?: (error: Error) => void
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
export async function startClientWakeCapture(
  options: ClientWakeCaptureOptions
): Promise<ClientWakeCaptureHandle> {
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

  const context = new AudioContextCtor()
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
  const queue: Float32Array[] = []
  let draining = false

  const drainQueue = async () => {
    if (draining) {
      return
    }
    draining = true
    try {
      while (!stopped && queue.length > 0) {
        const frame = queue.shift()
        if (!frame) {
          break
        }
        try {
          const pcm = floatToInt16LE(frame)
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
