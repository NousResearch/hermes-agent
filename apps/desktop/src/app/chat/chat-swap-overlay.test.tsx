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

  it('animates the glyph without any timer', () => {
    const { container } = render(<ChatSwapOverlay profile="turqoise" />)

    expect(container.querySelector('.glyph-spinner__strip')).toBeTruthy()
    expect(vi.getTimerCount()).toBe(0)

    vi.advanceTimersByTime(5_000)

    expect(vi.getTimerCount()).toBe(0)
  })

  it('names the waking profile', () => {
    render(<ChatSwapOverlay profile="turqoise" />)

    expect(screen.getByText(/turqoise/)).toBeTruthy()
  })

  it('conceals the interim session shell and raw profile name in Bot Mode', () => {
    const { container } = render(
      <div>
        <span>New session</span>
        <ChatSwapOverlay botMode profile="persephone" />
      </div>
    )

    const overlay = container.querySelector('[data-slot="chat-swap-overlay"]')

    expect(overlay?.className).toContain('bg-(--ui-chat-surface-background)')
    expect(screen.queryByText(/persephone/i)).toBeNull()
    expect(screen.getByText(/Bot Chat/i)).toBeTruthy()
  })

  it('keeps the last profile name through the fade-out, with the glyph frozen', () => {
    const { container, rerender } = render(<ChatSwapOverlay profile="turqoise" />)

    expect(container.querySelector('.glyph-spinner')?.hasAttribute('data-paused')).toBe(false)

    rerender(<ChatSwapOverlay profile={null} />)

    // Label held so the overlay doesn't blank while it fades.
    expect(screen.getByText(/turqoise/)).toBeTruthy()
    // ...and the spinner stops, the way clearing the interval used to stop it.
    expect(container.querySelector('.glyph-spinner')?.getAttribute('data-paused')).toBe('true')
  })
})
