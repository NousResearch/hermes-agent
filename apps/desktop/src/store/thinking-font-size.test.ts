import { describe, expect, it } from 'vitest'

import {
  normalizeThinkingFontSize,
  THINKING_FONT_SIZE_DEFAULT,
  THINKING_FONT_SIZE_MAX,
  THINKING_FONT_SIZE_MIN,
  thinkingFontSizeCodec
} from './thinking-font-size'

describe('normalizeThinkingFontSize', () => {
  it('keeps values inside the supported range', () => {
    expect(normalizeThinkingFontSize(13)).toBe(13)
  })

  it('clamps and rounds persisted values', () => {
    expect(normalizeThinkingFontSize(10)).toBe(THINKING_FONT_SIZE_MIN)
    expect(normalizeThinkingFontSize(16)).toBe(THINKING_FONT_SIZE_MAX)
    expect(normalizeThinkingFontSize('12.6')).toBe(13)
  })

  it('falls back to the shipped size for invalid values', () => {
    expect(normalizeThinkingFontSize('not-a-number')).toBe(THINKING_FONT_SIZE_DEFAULT)
  })

  it('persists the normalized size as a string', () => {
    expect(thinkingFontSizeCodec.encode(12.6)).toBe('13')
  })
})
