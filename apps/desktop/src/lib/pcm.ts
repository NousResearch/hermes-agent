/**
 * 16 kHz PCM helpers for live dictation.
 *
 * The capture pipeline resamples the mic to 16 kHz mono s16le in JS, so it
 * never depends on the browser granting an AudioContext at exactly 16 kHz
 * (macOS/Chromium can clamp the context rate to the device rate — a silent
 * rate mismatch that starved the streaming leg before the resampler existed).
 */

/** Resample mono Float32 samples (in -1..1) at *inputRate* to 16 kHz. */
export function resampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === 16000 || input.length === 0) {
    return input
  }

  const ratio = inputRate / 16000
  const outLength = Math.max(1, Math.floor(input.length / ratio))
  const output = new Float32Array(outLength)

  // Linear interpolation — handles integer (48k→16k) and fractional
  // (44.1k→16k) ratios alike.
  for (let i = 0; i < outLength; i += 1) {
    const pos = i * ratio
    const index = Math.floor(pos)
    const frac = pos - index

    if (index + 1 >= input.length) {
      output[i] = input[index]
    } else {
      output[i] = input[index] * (1 - frac) + input[index + 1] * frac
    }
  }

  return output
}

/** Convert mono Float32 samples to a 16 kHz s16le ArrayBuffer. */
export function float32ToInt16Pcm(input: Float32Array): ArrayBuffer {
  const pcm = new Int16Array(input.length)

  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]))
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
  }

  return pcm.buffer
}
