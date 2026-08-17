import { describe, expect, it } from 'vitest'

import { directiveFrameHeight } from './inline-preview-directive'

describe('directiveFrameHeight', () => {
  it('defaults when absent or garbage', () => {
    expect(directiveFrameHeight(undefined)).toBe(280)
    expect(directiveFrameHeight('')).toBe(280)
    expect(directiveFrameHeight('tall')).toBe(280)
    expect(directiveFrameHeight('12.5')).toBe(280)
  })

  it('clamps to the sane band', () => {
    expect(directiveFrameHeight('50')).toBe(120)
    expect(directiveFrameHeight('480')).toBe(480)
    expect(directiveFrameHeight('99999')).toBe(1200)
  })
})
