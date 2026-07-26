import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { Tip } from './tooltip'

// `Tip` mounts Radix lazily (on first hover/focus) because ~107 eager
// providers dominated unrelated interactions — dragging the sidebar splitter
// measured 105k TooltipProvider renders across a 60-frame gesture. The saving
// is only worth anything if the tip still actually appears, so this asserts
// the behavior the optimization has to preserve, not its implementation.

describe('Tip lazy mount', () => {
  it('shows the tooltip on hover even though Radix mounts only once armed', async () => {
    render(
      <Tip label="Reveal in sidebar">
        <button type="button">target</button>
      </Tip>
    )

    const trigger = screen.getByRole('button', { name: 'target' })

    expect(screen.queryByText('Reveal in sidebar')).toBeNull()

    // Arms the lazy wrapper, then drives Radix's own trigger handlers for the
    // same hover — the pointer is still inside, so the tip must open.
    fireEvent.pointerEnter(trigger)
    fireEvent.pointerMove(trigger)

    await waitFor(() => {
      expect(screen.getAllByText('Reveal in sidebar').length).toBeGreaterThan(0)
    })
  })

  it('renders the child untouched when there is no label', () => {
    const { container } = render(
      <Tip label={undefined}>
        <button type="button">bare</button>
      </Tip>
    )

    expect(screen.getByRole('button', { name: 'bare' })).toBeTruthy()
    expect(container.querySelector('[data-slot="tooltip-idle"]')).toBeNull()
  })

  it('does not mount Radix until the pointer arrives', () => {
    const { container } = render(
      <Tip label="Later">
        <button type="button">idle</button>
      </Tip>
    )

    // The idle wrapper is present; no Radix trigger has been created.
    expect(container.querySelector('[data-slot="tooltip-idle"]')).toBeTruthy()
    expect(container.querySelector('[data-slot="tooltip-trigger"]')).toBeNull()
  })
})
