import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { Tip } from './tooltip'

describe('Tip', () => {
  afterEach(() => {
    cleanup()
  })

  it('shows on pointer enter and dismisses on pointer leave', async () => {
    render(
      <Tip label="Layout editor — ⌘-click resets the layout">
        <button type="button">layout</button>
      </Tip>
    )

    const trigger = screen.getByRole('button', { name: 'layout' })

    fireEvent.pointerMove(trigger, { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toContain(
      'Layout editor — ⌘-click resets the layout'
    )

    fireEvent.pointerLeave(trigger)
    await waitFor(() => {
      expect(screen.queryByRole('tooltip')).toBeNull()
    })
  })

  it('never captures pointer events on the tip surface', async () => {
    render(
      <Tip label="Blocked?">
        <button type="button">target</button>
      </Tip>
    )

    fireEvent.pointerMove(screen.getByRole('button', { name: 'target' }), { pointerType: 'mouse' })
    const tip = await screen.findByRole('tooltip')
    // Role lives on the visually-hidden a11y node; the portaled content root
    // is the data-slot wrapper that must stay click-through.
    const content = tip.closest('[data-slot="tooltip-content"]') ?? tip.parentElement
    expect(content?.className).toMatch(/pointer-events-none/)
  })

  it('renders the child alone when label is empty', () => {
    render(
      <Tip label="">
        <button type="button">bare</button>
      </Tip>
    )

    expect(screen.getByRole('button', { name: 'bare' })).toBeTruthy()
    expect(screen.queryByRole('tooltip')).toBeNull()
  })
})
