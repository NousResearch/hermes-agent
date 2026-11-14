import type * as React from 'react'

import { cn } from '@/lib/utils'

// A 4x4 cell checkerboard — the app's "dither" marker — drawn as real rects
// instead of the `.dither` CSS gradient.
//
// Why the gradient fails here and nowhere else: `.dither` paints 2px cells with
// `repeating-conic-gradient`, which Chromium rasterizes on the *device* grid.
// At a display scale that isn't a whole multiple of 100% (Windows 125/150/175%
// are the common ones) a 2px cell is 2.5-3.5 device pixels, so every cell edge
// lands mid-pixel and gets antialiased. An 8px marker only holds 4x4 cells, so
// the AA edges are most of the ink and the dot stops reading as a dither — it
// reads as a smudge next to the section title. `shape-rendering="crispEdges"`
// snaps rect edges to whole device pixels at any scale, so the grid survives.
//
// Sized in em-free `size-*` utilities so callers can override (size-3,
// text-primary, a blink animation) exactly as they did on the `.dither` span.
export function DitherDot({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span aria-hidden="true" className={cn('inline-block size-2 shrink-0', className)} {...props}>
      <svg className="size-full" shapeRendering="crispEdges" viewBox="0 0 8 8" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M0 0h2v2H0zM4 0h2v2H4zM2 2h2v2H2zM6 2h2v2H6zM0 4h2v2H0zM4 4h2v2H4zM2 6h2v2H2zM6 6h2v2H6z"
          fill="currentColor"
        />
      </svg>
    </span>
  )
}
