// The overlay used to run its own 80ms setInterval + setState braille ticker —
// the same mechanism class (per-tick DOM mutation scheduling a style recalc)
// that GlyphSpinner was rewritten to remove. It now renders GlyphSpinner, so
// what needs pinning is that no timer comes back, that the label still survives
// the fade-out, and that the spinner stops animating once the swap is done.
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ChatSwapOverlay } from './chat-swap-overlay'

afterEach(() => {
  cleanup()
})

describe('ChatSwapOverlay', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('keeps the last profile name through the fade-out, with the glyph frozen', () => {
    const { container, rerender } = render(<ChatSwapOverlay profile="turqoise" />)

    const overlay = container.firstElementChild
    expect(overlay?.classList.contains('opacity-100')).toBe(true)
    expect(overlay?.classList.contains('opacity-0')).toBe(false)
    expect(container.querySelector('.glyph-spinner')?.hasAttribute('data-paused')).toBe(false)

    rerender(<ChatSwapOverlay profile={null} />)

    // Label held so the overlay doesn't blank while it fades.
    expect(screen.getByText(/turqoise/)).toBeTruthy()
    expect(container.firstElementChild).toBe(overlay)
    expect(overlay?.classList.contains('opacity-100')).toBe(false)
    expect(overlay?.classList.contains('opacity-0')).toBe(true)
    // ...and the spinner stops, the way clearing the interval used to stop it.
    expect(container.querySelector('.glyph-spinner')?.getAttribute('data-paused')).toBe('true')
  })
})
