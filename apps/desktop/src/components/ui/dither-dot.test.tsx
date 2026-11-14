// The section marker must stay a marker at fractional display scales.
//
// `.dither` (styles.css) paints its 2px cells from a `repeating-conic-gradient`,
// which Chromium rasterizes on the device grid: at 125/150/175% Windows scaling
// each cell edge is a fraction of a pixel, so an 8px dot is mostly antialiased
// edge and reads as a smudge beside the title. The replacement draws real rects
// with shape-rendering="crispEdges", which snaps to whole device pixels. Guard
// the two properties that make that true, plus the sizing contract callers rely
// on (`size-2` default, overridable).
import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { DitherDot } from './dither-dot'

afterEach(cleanup)

it('draws real rects with crispEdges instead of the CSS gradient', () => {
  const { container } = render(<DitherDot />)
  const svg = container.querySelector('svg')

  expect(svg).not.toBeNull()
  expect(svg?.getAttribute('shape-rendering')).toBe('crispEdges')
  // 4x4 checkerboard => half the cells painted.
  expect(svg?.querySelectorAll('path')).toHaveLength(1)
  expect(container.querySelector('.dither')).toBeNull()
})

it('stays decorative', () => {
  const { container } = render(<DitherDot />)

  expect(container.firstElementChild?.getAttribute('aria-hidden')).toBe('true')
})

it('keeps the 8px default and lets callers resize/recolor it', () => {
  const { container } = render(<DitherDot className="size-3 text-primary" />)
  const className = container.firstElementChild?.className ?? ''

  expect(className).toContain('size-3')
  expect(className).not.toContain('size-2')
  expect(className).toContain('text-primary')
  // The svg fills its box, so an overridden size keeps the grid square.
  expect(container.querySelector('svg')?.getAttribute('class')).toContain('size-full')
})
