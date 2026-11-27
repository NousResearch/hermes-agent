import '../styles.css'

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { StatusDot } from './status-dot'

afterEach(cleanup)

describe('StatusDot', () => {
  it('uses a bounded breath for the good tone', () => {
    const { container } = render(<StatusDot tone="good" />)
    const dot = container.firstElementChild

    expect(dot).toBeInstanceOf(HTMLElement)
    expect(dot?.classList.contains('status-dot-breath')).toBe(true)
    expect((dot as HTMLElement).style.animationIterationCount).toBe('2')
  })

  it('keeps non-good tones static', () => {
    const { container } = render(<StatusDot tone="muted" />)
    const dot = container.firstElementChild

    expect(dot?.classList.contains('status-dot-breath')).toBe(false)
  })
})
