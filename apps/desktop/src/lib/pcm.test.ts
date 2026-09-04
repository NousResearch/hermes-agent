import { describe, expect, it } from 'vitest'

import { float32ToInt16Pcm, resampleTo16k } from './pcm'

describe('resampleTo16k', () => {
  it('passes 16 kHz input through unchanged', () => {
    const input = new Float32Array([0.1, -0.2, 0.3])

    expect(resampleTo16k(input, 16000)).toBe(input)
  })

  it('downsamples integer ratios (48k -> 16k) by 3x', () => {
    // One second of a 1 kHz-ish ramp at 48 kHz.
    const input = new Float32Array(48000)

    for (let i = 0; i < input.length; i += 1) {
      input[i] = Math.sin((2 * Math.PI * 1000 * i) / 48000)
    }

    const out = resampleTo16k(input, 48000)

    expect(out.length).toBe(16000)

    // The downsampled signal must still be a sine at 1 kHz: value at sample
    // n equals the 48 kHz sine at 3n (linear interp on an exact multiple).
    for (const n of [0, 1000, 8000, 15999]) {
      expect(out[n]).toBeCloseTo(input[n * 3], 4)
    }
  })

  it('handles fractional ratios (44.1k -> 16k) without crashing or NaN', () => {
    const input = new Float32Array(44100)

    for (let i = 0; i < input.length; i += 1) {
      input[i] = Math.sin((2 * Math.PI * 440 * i) / 44100)
    }

    const out = resampleTo16k(input, 44100)

    expect(out.length).toBe(Math.floor(44100 / (44100 / 16000)))
    expect(out.every(value => Number.isFinite(value))).toBe(true)
    expect(out.every(value => value >= -1.001 && value <= 1.001)).toBe(true)
  })

  it('returns empty for empty input', () => {
    expect(resampleTo16k(new Float32Array(0), 48000).length).toBe(0)
  })
})

describe('float32ToInt16Pcm', () => {
  it('converts and clamps', () => {
    const buffer = float32ToInt16Pcm(new Float32Array([0, 1, -1, 2, -2]))
    const pcm = new Int16Array(buffer)

    expect(pcm[0]).toBe(0)
    expect(pcm[1]).toBe(32767)
    expect(pcm[2]).toBe(-32768)
    expect(pcm[3]).toBe(32767)
    expect(pcm[4]).toBe(-32768)
  })
})
