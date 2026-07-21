import { describe, expect, it } from 'vitest'

import { splitSpeechText } from './speech-text'

describe('splitSpeechText', () => {
  it('splits long text on natural boundaries without losing words', () => {
    const text = Array.from(
      { length: 80 },
      (_, index) => `Sentence ${index + 1} explains one part of the recommendation clearly.`
    ).join(' ')

    const chunks = splitSpeechText(text, 320)

    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.every(chunk => chunk.length <= 320)).toBe(true)
    expect(chunks.join(' ')).toBe(text)
  })

  it('hard-splits a single overlong unit while preserving all text', () => {
    const text = 'x'.repeat(725)

    const chunks = splitSpeechText(text, 300)

    expect(chunks).toEqual(['x'.repeat(300), 'x'.repeat(300), 'x'.repeat(125)])
    expect(chunks.join('')).toBe(text)
  })
})
