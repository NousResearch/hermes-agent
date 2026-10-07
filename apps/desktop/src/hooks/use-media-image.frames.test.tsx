import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useMediaImage } from './use-media-image'

// Regression for #74564 (aperture half): the old preview clamp
// (--image-preview-height: clamp(16.25rem, …, 26.25rem), max-width 34rem)
// meant the reserved frame for a landscape image could never span the
// preview width — a 16:9 frame was at most 26.25rem × 16/9 ≈ 46.7rem, and
// on typical windows the viewport clamp pinned it nearer 16.25rem × 16/9 ≈
// 28.9rem, so stacked landscape images showed through a small aperture
// (the reporter measured ~10% of one image's height).
//
// The envelope lives in styles.css as two custom properties; both the
// useMediaImage frame width and the unframed max-h in markdown-text.tsx
// derive from them. Pin the contract at that seam.
describe('inline image preview envelope (#74564)', () => {
  const stylesheet = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), '../styles.css'), 'utf8')

  beforeEach(() => {
    vi.stubGlobal('window', { hermesDesktop: {} })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const maxWidth = Number(stylesheet.match(/--image-preview-max-width:\s*([\d.]+)rem/)?.[1])
  const heightClamp = stylesheet.match(
    /--image-preview-height:\s*clamp\(([\d.]+)rem,\s*calc\(var\(--vsq\)\s*\*\s*([\d.]+)\),\s*([\d.]+)rem\)/
  )
  const [floor, vsqFactor, ceiling] = (heightClamp?.slice(1) ?? []).map(Number)

  it('guarantees a landscape frame can span the full preview width', () => {
    expect(maxWidth).not.toBeNaN()
    expect(heightClamp).not.toBeNull()

    // The height FLOOR must be at least maxWidth × 9/16: then a 16:9 frame
    // (width = height × 16/9) reaches the full preview width even on the
    // shortest viewport, instead of collapsing to the old 16.25rem floor.
    expect(floor).toBeGreaterThanOrEqual((maxWidth * 9) / 16)
    // And the envelope actually grew from the reported 34rem/16.25rem pair.
    expect(maxWidth).toBeGreaterThan(34)
    expect(floor).toBeGreaterThan(16.25)
  })

  it('keeps growing with the viewport in tall windows', () => {
    // The middle clamp term scales with --vsq (min(0.5vh, 0.5vw)), so the
    // preview tracks the window's shorter side between floor and ceiling.
    // A factor at or below the old 100 — or a ceiling back at the old
    // 26.25rem cap — would quietly shrink previews in tall windows again.
    expect(vsqFactor).toBeGreaterThan(100)
    expect(ceiling).toBeGreaterThan(26.25)
    expect(ceiling).toBeGreaterThan(floor)
    // Portrait images lean on the ceiling: it must at least match the width
    // cap so a 1:1 frame can also fill the preview on a tall enough window.
    expect(ceiling).toBeGreaterThanOrEqual(maxWidth)
  })

  it('keeps the natural-size cap so a small image is never upscaled', () => {
    const { result } = renderHook(() =>
      useMediaImage('/home/user/out/shot.png', 16 / 9, { width: 640, height: 360 })
    )

    expect(String(result.current.frameStyle?.width)).toContain('640px')
  })
})
