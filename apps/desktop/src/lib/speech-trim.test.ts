import { describe, expect, it } from 'vitest'

import { decodeAudioClip, speechRange } from './speech-trim'

/**
 * Build a decoded-clip stand-in. The silence map mirrors the padding measured
 * against the real endpoint on 2026-11-09 (gpt-4o-mini-tts + marin, read with
 * `ffmpeg -af silencedetect=noise=-40dB:d=0.12`): ~0.36 s of dead head, a short
 * gap between sentences, and a tail after the last mark.
 */
function stubBuffer(duration: number, spans: Array<[number, number]>, sampleRate = 24000) {
  const length = Math.floor(duration * sampleRate)
  const data = new Float32Array(length).fill(0)

  for (const [from, to] of spans) {
    for (let i = Math.floor(from * sampleRate); i < Math.min(length, Math.floor(to * sampleRate)); i += 1) {
      data[i] = 0.2
    }
  }

  return {
    duration,
    getChannelData: () => data,
    length,
    numberOfChannels: 1,
    sampleRate
  } as unknown as AudioBuffer
}

describe('speechRange', () => {
  it('drops the provider dead head and keeps a little tail air', () => {
    // 3.60 s clip: silence 0.524-0.888, 1.921-2.290, 3.278-3.600.
    const buffer = stubBuffer(3.6, [
      [0.888, 1.921],
      [2.29, 3.278]
    ])
    const range = speechRange(buffer)

    expect(range.trimmedHead).toBeGreaterThan(0.8)
    expect(range.trimmedHead).toBeLessThan(0.95)
    expect(range.start + range.duration).toBeGreaterThan(3.25)
    // 3.278 (last speech) - 0.888 (first speech) = 2.39, plus the 0.06 tail
    // keep and one 5 ms analysis window of granularity at each end.
    expect(range.duration).toBeLessThan(2.47)
  })

  it('reports how much padding it removed', () => {
    const buffer = stubBuffer(2.0, [[0.4, 1.6]])
    const range = speechRange(buffer)

    expect(range.trimmedHead).toBeCloseTo(0.4, 2)
    expect(range.trimmedTail).toBeCloseTo(0.34, 1)
  })

  it('keeps an all-silent clip intact instead of collapsing it', () => {
    // A provider silence hallucination must not become a zero-length element:
    // that fires `ended` immediately and turns a seam into a click.
    const buffer = stubBuffer(1.0, [])
    const range = speechRange(buffer)

    expect(range.start).toBe(0)
    expect(range.duration).toBeCloseTo(1.0, 2)
  })

  it('never reports a zero-length range', () => {
    const buffer = stubBuffer(0.5, [[0.2, 0.21]])
    const range = speechRange(buffer)

    expect(range.duration).toBeGreaterThanOrEqual(0.01)
  })
})

describe('decodeAudioClip', () => {
  it('returns null instead of throwing when the bytes cannot be decoded', async () => {
    const context = { decodeAudioData: async () => Promise.reject(new Error('no decoder')) } as unknown as AudioContext

    await expect(decodeAudioClip(new ArrayBuffer(8), context)).resolves.toBeNull()
  })
})
