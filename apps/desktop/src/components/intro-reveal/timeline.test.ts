import { describe, expect, it } from 'vitest'

import {
  beatsBetween,
  INTRO_BEATS,
  INTRO_DEADMAN_MS,
  INTRO_PROMPT,
  INTRO_REPLY_WORDS,
  INTRO_TOOL_ROWS,
  INTRO_TOTAL_MS,
  sampleCurves,
  streamingSchedule,
  typingSchedule
} from './timeline'

describe('intro timeline', () => {
  it('beats are strictly ordered by time', () => {
    for (let i = 1; i < INTRO_BEATS.length; i += 1) {
      expect(INTRO_BEATS[i].t).toBeGreaterThan(INTRO_BEATS[i - 1].t)
    }
  })

  it('typing finishes before send', () => {
    const sendT = INTRO_BEATS.find(b => b.id === 'send')!.t
    const times = typingSchedule(INTRO_PROMPT, 700, sendT - 450)

    expect(times.length).toBe(INTRO_PROMPT.length)
    expect(times[times.length - 1]).toBeLessThan(sendT)

    for (let i = 1; i < times.length; i += 1) {
      expect(times[i]).toBeGreaterThan(times[i - 1])
    }
  })

  it('typing cadence is deterministic', () => {
    const a = typingSchedule(INTRO_PROMPT, 0, 1000)
    const b = typingSchedule(INTRO_PROMPT, 0, 1000)

    expect(a).toEqual(b)
  })

  it('tool rows appear after send and complete before the reply lands', () => {
    const sendT = INTRO_BEATS.find(b => b.id === 'send')!.t
    const replyT = INTRO_BEATS.find(b => b.id === 'reply')!.t

    for (const row of INTRO_TOOL_ROWS) {
      expect(row.at).toBeGreaterThan(sendT)
      expect(row.doneAt).toBeGreaterThan(row.at)
      expect(row.doneAt).toBeLessThan(replyT)
    }
  })

  it('reply streaming is monotonic and fits between reply and everywhere', () => {
    const replyT = INTRO_BEATS.find(b => b.id === 'reply')!.t
    const everywhereT = INTRO_BEATS.find(b => b.id === 'everywhere')!.t
    const times = streamingSchedule(INTRO_REPLY_WORDS.length, replyT + 150, replyT + 2400)

    expect(times.length).toBe(INTRO_REPLY_WORDS.length)
    expect(times[times.length - 1]).toBeLessThanOrEqual(everywhereT)

    for (let i = 1; i < times.length; i += 1) {
      expect(times[i]).toBeGreaterThanOrEqual(times[i - 1])
    }
  })

  it('glow blooms for the brand close and fades by the end', () => {
    const brandT = INTRO_BEATS.find(b => b.id === 'brand')!.t

    expect(sampleCurves(brandT + 1400).glow).toBeGreaterThan(0.9)
    expect(sampleCurves(INTRO_TOTAL_MS + 500).glow).toBeLessThan(0.15)
  })

  it('scatter completes by the end and curves stay in [0,1]', () => {
    expect(sampleCurves(INTRO_TOTAL_MS).scatter).toBeGreaterThan(0.9)

    for (let t = 0; t <= INTRO_TOTAL_MS + 1000; t += 50) {
      const c = sampleCurves(t)

      for (const v of [c.glow, c.scatter]) {
        expect(v).toBeGreaterThanOrEqual(0)
        expect(v).toBeLessThanOrEqual(1)
      }
    }
  })

  it('deadman exceeds the sequence length', () => {
    expect(INTRO_DEADMAN_MS).toBeGreaterThan(INTRO_TOTAL_MS)
  })

  it('fires each beat exactly once across irregular frame steps', () => {
    const fired = new Set<string>()
    let prev = -1
    let t = 0

    while (t < INTRO_TOTAL_MS) {
      t = Math.min(INTRO_TOTAL_MS, t + 33 + Math.random() * 8)

      for (const b of beatsBetween(prev, t)) {
        expect(fired.has(b.id)).toBe(false)
        fired.add(b.id)
      }

      prev = t
    }

    expect(fired.size).toBe(INTRO_BEATS.length)
  })
})
