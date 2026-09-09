'use client'

import { useMemo } from 'react'

import { sanitizeSvgMarkup } from '@/lib/svg-sanitize'

import type { RichFenceProps } from './types'

// Lazy chunk (pulls in DOMPurify). Renders a ```svg fence as an image after
// sanitizing markup and applying the stricter no-resource SVG policy.
export default function SvgRenderer({ code }: RichFenceProps) {
  const clean = useMemo(() => sanitizeSvgMarkup(code), [code])

  if (!clean.trim()) {
    return null
  }

  // Left-aligned, capped on both axes so a large intrinsic SVG scales down
  // (preserving ratio) instead of filling the column or centering.
  return (
    <div
      className="my-2 [&_svg]:block [&_svg]:h-auto [&_svg]:w-auto [&_svg]:max-h-[33dvh] [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: clean }}
    />
  )
}
