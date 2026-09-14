// GlyphSpinner animates on the compositor: every frame is in the DOM from
// mount and a transform keyframes animation scrolls between them. It has no
// timer and performs no per-frame DOM write, because the setInterval +
// `glyph.textContent` ticker this replaced was scheduling a document-scale
// style recalculation on every tick.
//
// These tests therefore pin the DATA and WIRING that make the CSS correct,
// which is all jsdom can see — it has no animation engine, so the motion
// itself is not observable here:
//
//  - the strip carries every frame, in order, so `steps(N)` lands on each one;
//  - the custom properties feeding `steps()` / duration match the source data,
//    so cadence per variant is preserved;
//  - no timer is ever created (the ticker is really gone);
//  - the kept-alive-tab gate still resolves to a paused animation.
//
// The CSS itself is verified where it can actually run: e2e/glyph-spinner.spec.ts
// reads getComputedStyle / getAnimations in a real browser. Asserting on the
// text of the stylesheet from here would test the shape of the source rather
// than its behaviour (banned outright — see AGENTS.md), and did in fact break
// on a pure var()-fallback edit that changed no rendered pixel.
import { render, screen } from '@testing-library/react'
import { Profiler, type ProfilerOnRenderCallback } from 'react'
import spinners from 'unicode-animations'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { GlyphSpinner } from './glyph-spinner'

const BRAILLE = spinners.braille

function strip(): HTMLElement {
  const status = screen.getByRole('status', { name: 'Loading' })
  const found = status.querySelector<HTMLElement>('.glyph-spinner__strip')

  if (!found) {
    throw new Error('no frame strip rendered')
  }

  return found
}

describe('GlyphSpinner', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('renders every frame in source order as the scroll strip', () => {
    render(<GlyphSpinner spinner="braille" />)

    const frames = [...strip().querySelectorAll('.glyph-spinner__frame')].map(node => node.textContent)

    expect(frames).toEqual([...BRAILLE.frames])
    // The old ticker started on frame 0; steps() starts the strip there too.
    expect(frames[0]).toBe('⠋')
  })

  it('feeds steps() and the duration from the spinner data, so cadence is unchanged', () => {
    render(<GlyphSpinner spinner="braille" />)

    const style = strip().style

    // One full cycle is frames x interval; steps(frames) parks on each frame
    // for exactly `interval` ms, which is what the setInterval did.
    expect(style.getPropertyValue('--glyph-spinner-frames')).toBe(String(BRAILLE.frames.length))
    expect(style.getPropertyValue('--glyph-spinner-duration')).toBe(`${BRAILLE.frames.length * BRAILLE.interval}ms`)
  })

  it('keeps per-variant cadence distinct', () => {
    render(<GlyphSpinner ariaLabel="Working" spinner="breathe" />)

    const node = screen.getByRole('status', { name: 'Working' })
    const found = node.querySelector<HTMLElement>('.glyph-spinner__strip')!
    const breathe = spinners.breathe

    expect(found.querySelectorAll('.glyph-spinner__frame')).toHaveLength(breathe.frames.length)
    expect(found.style.getPropertyValue('--glyph-spinner-duration')).toBe(
      `${breathe.frames.length * breathe.interval}ms`
    )
  })

  it('creates no timer and no update-phase React commit', () => {
    let updateCommits = 0

    const onRender: ProfilerOnRenderCallback = (_id, phase) => {
      if (phase !== 'mount') {
        updateCommits += 1
      }
    }

    render(
      <Profiler id="glyph-spinner" onRender={onRender}>
        <GlyphSpinner spinner="braille" />
      </Profiler>
    )

    // The whole point: the ticker is gone, so nothing is scheduled at all.
    expect(vi.getTimerCount()).toBe(0)

    vi.advanceTimersByTime(5_000)

    expect(vi.getTimerCount()).toBe(0)
    expect(updateCommits).toBe(0)
  })

  it('pauses while its kept-alive pane is hidden, and resumes when shown', () => {
    const { rerender } = render(
      <PaneVisibleContext.Provider value={false}>
        <GlyphSpinner spinner="braille" />
      </PaneVisibleContext.Provider>
    )

    const viewport = () => screen.getByRole('status', { name: 'Loading' }).querySelector('.glyph-spinner')

    expect(viewport()?.getAttribute('data-paused')).toBe('true')

    rerender(
      <PaneVisibleContext.Provider value>
        <GlyphSpinner spinner="braille" />
      </PaneVisibleContext.Provider>
    )

    expect(viewport()?.hasAttribute('data-paused')).toBe(false)
  })

  it('hides the decorative frames from assistive tech', () => {
    render(<GlyphSpinner spinner="braille" />)

    // role="status" is a live region. The frames must not be announced — the
    // old implementation rewrote this region's text ~12x/second.
    const status = screen.getByRole('status', { name: 'Loading' })

    expect(status.querySelector('.glyph-spinner')?.getAttribute('aria-hidden')).toBe('true')
  })

  it('pauses on request while staying mounted', () => {
    // For a caller that keeps the spinner in the tree through a fade-out
    // (ChatSwapOverlay) and does not want it animating once the wait is over.
    const { rerender } = render(<GlyphSpinner paused spinner="braille" />)
    const viewport = () => screen.getByRole('status', { name: 'Loading' }).querySelector('.glyph-spinner')

    expect(viewport()?.getAttribute('data-paused')).toBe('true')

    rerender(<GlyphSpinner paused={false} spinner="braille" />)

    expect(viewport()?.hasAttribute('data-paused')).toBe(false)
  })
})
