import { describe, expect, it } from 'vitest'

import { formatRefValue, hermesDirectiveFormatter } from './directive-text'

describe('formatRefValue', () => {
  it('leaves simple paths untouched', () => {
    expect(formatRefValue('src/index.ts')).toBe('src/index.ts')
    expect(formatRefValue('https://example.com/post')).toBe('https://example.com/post')
  })

  it('wraps paths with whitespace in backticks', () => {
    expect(formatRefValue('apple-touch-icon (1).png')).toBe('`apple-touch-icon (1).png`')
  })

  it('falls back to double quotes when value contains backticks', () => {
    expect(formatRefValue('weird `name` (1).md')).toBe('"weird `name` (1).md"')
  })
})

describe('hermesDirectiveFormatter.parse', () => {
  it('keeps quoted file paths whole when parsing', () => {
    const segments = hermesDirectiveFormatter.parse('see @image:`apple-touch-icon (1).png` for the icon')

    expect(segments).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'mention', type: 'image', label: 'apple-touch-icon (1).png', id: 'apple-touch-icon (1).png' },
      { kind: 'text', text: ' for the icon' }
    ])
  })

  it('still parses unquoted paths', () => {
    const segments = hermesDirectiveFormatter.parse('@file:src/main.tsx the entry point')

    expect(segments).toEqual([
      { kind: 'mention', type: 'file', label: 'main.tsx', id: 'src/main.tsx' },
      { kind: 'text', text: ' the entry point' }
    ])
  })
})
