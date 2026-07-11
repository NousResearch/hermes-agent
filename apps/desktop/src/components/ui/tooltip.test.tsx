import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { Tip, TipHintLabel } from './tooltip'

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

  it('forces a block-level label child back to inline-flex via the decoration wrapper class', async () => {
    render(
      <Tip label={<span className="flex items-center gap-2">broken label</span>}>
        <button type="button">trigger</button>
      </Tip>
    )

    fireEvent.pointerMove(screen.getByRole('button', { name: 'trigger' }), { pointerType: 'mouse' })
    await screen.findByRole('tooltip')

    // jsdom doesn't apply real Tailwind CSS, so getComputedStyle can't prove
    // the arbitrary-variant class takes effect. Assert the guarding class is
    // present on the decoration wrapper instead — that's what actually forces
    // any direct child (including the flex span above) to render inline-flex
    // in a real browser. See #62022.
    const content = document.querySelector<HTMLElement>('[data-slot="tooltip-content"]')
    const decoration = content?.firstElementChild

    expect(decoration?.className).toMatch(/\[&>\*\]:!inline-flex/)
  })

  it('TipHintLabel renders inline-flex with a hint, plain text without one', () => {
    const { rerender } = render(<TipHintLabel hint="Ctrl+`" text="PowerShell" />)
    const withHint = screen.getByText('PowerShell').parentElement
    expect(withHint?.classList.contains('inline-flex')).toBe(true)
    expect(withHint?.classList.contains('flex')).toBe(false)

    rerender(<TipHintLabel text="PowerShell" />)
    expect(screen.queryByText('PowerShell')?.tagName).not.toBe('SPAN')
  })
})
