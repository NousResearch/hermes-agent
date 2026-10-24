'use client'

import { useMemo } from 'react'

import { svgImageSrc } from '@/lib/svg-image'

import type { RichFenceProps } from './types'

// Renders a ```svg fence as an <img>: no script runs, no outside URL loads,
// and the SVG's own <style>/ids cannot reach the app (see svgImageSrc).
export default function SvgRenderer({ code }: RichFenceProps) {
  const src = useMemo(() => svgImageSrc(code), [code])

  if (!src) {
    return null
  }

  // Left-aligned, capped on both axes so a large intrinsic SVG scales down
  // (preserving ratio) instead of filling the column or centering.
  return <img alt="" className="my-2 block h-auto max-h-[33dvh] w-auto max-w-full" src={src} />
}
