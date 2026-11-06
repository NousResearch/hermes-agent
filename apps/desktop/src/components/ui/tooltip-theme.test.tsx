import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { RootTooltipProvider, Tip } from './tooltip'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// The bubble used to paint itself with the inverted pair `bg-foreground
// text-background` — on a dark theme `--foreground` is near-white, so every
// tip on screen opened as a white bubble outside the theme (#140931). The
// bubble must use the themed floating surface menus and popovers ship (menu.ts
// `menuSurfaceClass`, popover.tsx): the elevated background token, its stroke,
// and the popover text colour — and the arrow must fill itself with the same
// surface instead of `--foreground`, or a themed bubble would carry a white
// spike. jsdom resolves no CSS, so the assertions read the utility classes
// themselves, exactly as authored.
it('paints the bubble with the themed floating surface, not the inverted pair', async () => {
  render(
    <RootTooltipProvider>
      <Tip label="Themed tip">
        <button>trigger</button>
      </Tip>
    </RootTooltipProvider>
  )

  const trigger = screen.getByRole('button')

  fireEvent.pointerEnter(trigger)
  fireEvent.pointerMove(trigger, { pointerType: 'mouse' })

  await vi.waitFor(() => {
    // eslint-disable-next-line no-restricted-globals -- the portal mounts on the live document
    const bubble = document.querySelector('[data-slot="tooltip-content"]')

    expect(bubble).not.toBeNull()
  })

  // eslint-disable-next-line no-restricted-globals -- the portal mounts on the live document
  const bubbleClass = document.querySelector('[data-slot="tooltip-content"]')?.className ?? ''

  expect(bubbleClass).toContain('bg-(--tooltip-surface)')
  expect(bubbleClass).toContain('[--tooltip-surface:color-mix(in_srgb,var(--ui-bg-elevated)_96%,transparent)]')
  expect(bubbleClass).toContain('border-(--ui-stroke-secondary)')
  expect(bubbleClass).toContain('text-popover-foreground')

  // The inverted pair is the regression: it must never come back.
  expect(bubbleClass).not.toContain('bg-foreground')
  expect(bubbleClass).not.toContain('text-background')

  // eslint-disable-next-line no-restricted-globals -- the portal mounts on the live document
  const arrow = document.querySelector('[data-slot="tooltip-arrow"]')

  expect(arrow).toBeInstanceOf(HTMLSpanElement)
  expect(arrow?.querySelector('span')?.className).toContain('bg-(--tooltip-surface)')
})
