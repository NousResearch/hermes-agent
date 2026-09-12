import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { CompressionCountStatus } from './compression-count-status'

afterEach(cleanup)

describe('CompressionCountStatus', () => {
  it('renders an explicit zero with a count tooltip', async () => {
    render(<CompressionCountStatus count={0} />)

    const counter = screen.getByTestId('compression-count')
    expect(counter.textContent).toContain('0')
    expect(counter.getAttribute('aria-label')).toBe('Compressions: 0')

    fireEvent.pointerMove(counter, { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toBe('Compressions: 0')
  })

  it('does not fabricate a count when live usage omits it', () => {
    const { container } = render(<CompressionCountStatus />)

    expect(container.innerHTML).toBe('')
  })
})
