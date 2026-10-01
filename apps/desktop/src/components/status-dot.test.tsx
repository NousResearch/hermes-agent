import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { StatusDot, type StatusTone } from './status-dot'

afterEach(cleanup)

const TONES: StatusTone[] = ['good', 'warn', 'bad', 'muted']

function mark(tone: StatusTone): HTMLElement {
  const { container } = render(<StatusDot tone={tone} />)
  const element = container.firstElementChild

  if (!(element instanceof HTMLElement)) {
    throw new Error('status mark did not render')
  }

  return element
}

describe('StatusDot', () => {
  it('gives every tone its own shape so state never rests on hue alone', () => {
    const shapes = TONES.map(tone => mark(tone).dataset.statusShape)

    expect(shapes.every(Boolean)).toBe(true)
    expect(new Set(shapes).size).toBe(TONES.length)
  })

  it('stays out of the accessibility tree; nearby copy carries the state', () => {
    expect(mark('warn').getAttribute('aria-hidden')).toBe('true')
  })
})
