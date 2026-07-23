import { type CSSProperties, useEffect, useRef } from 'react'

import eggSheetUrl from './pet-egg-sheet.png'

/**
 * Animated pixel egg — the iamcrog "bouncing hatching egg" 12-frame sheet
 * (32×32 cells, stacked vertically), drawn to a canvas and recolored to a warm
 * white/creme shell.
 *
 * The sheet's shell is mid-gray, so a plain multiply only darkens it (still
 * gray). Instead we remap each pixel's luminance through a creme ramp via a 256-
 * entry LUT: near-black stays a warm dark outline, midtones become creme shadow,
 * highlights go near-white. Done on a 32×32 offscreen then nearest-neighbor
 * scaled up so it stays crisp.
 *
 * Frames 0–5 are the intact squash/stretch bounce; 6–11 are the crack/hatch.
 * `mode="bounce"` loops 0–5 (never shows a crack); `mode="hatch"` plays 6–11
 * once then calls onDone.
 */

const FRAME = 32
const TOTAL_FRAMES = 12
const BOUNCE_FRAMES = 6 // 0..5 — intact egg only; cracks start at frame 6
const HATCH_START = 6 // first crack frame
// Per-frame speed *while* a bounce is playing.
const BOUNCE_MS = 250
const HATCH_MS = 190
// Harvest-Moon idle: the egg rests on frame 0 for a long, randomized gap between
// bounces so it reads as "occasionally stirs", not "constantly animating".
const REST_MIN_MS = 2600
const REST_MAX_MS = 6200

// Creme ramp endpoints: warm dark outline → creme shadow → near-white highlight.
const OUTLINE: [number, number, number] = [78, 66, 58]
const SHADOW: [number, number, number] = [214, 198, 168]
const HIGHLIGHT: [number, number, number] = [253, 249, 238]
const OUTLINE_CUTOFF = 46

const lerp = (a: number, b: number, t: number) => a + (b - a) * t

// Precompute the luminance→creme mapping once (shared across every egg). Below
// the cutoff it's the flat outline; above, a SHADOW→HIGHLIGHT ramp.
const CREME_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3)

  for (let g = 0; g < 256; g++) {
    const dark = g < OUTLINE_CUTOFF
    const t = dark ? 0 : (g - OUTLINE_CUTOFF) / (255 - OUTLINE_CUTOFF)
    const from = dark ? OUTLINE : SHADOW
    const to = dark ? OUTLINE : HIGHLIGHT
    lut.set([lerp(from[0], to[0], t), lerp(from[1], to[1], t), lerp(from[2], to[2], t)], g * 3)
  }

  return lut
})()

let _sheet: HTMLImageElement | null = null
let _sheetLoading: Promise<HTMLImageElement> | null = null

function loadSheet(): Promise<HTMLImageElement> {
  if (_sheet?.complete) {
    return Promise.resolve(_sheet)
  }

  if (!_sheetLoading) {
    _sheetLoading = new Promise((resolve, reject) => {
      const img = new Image()

      img.onload = () => {
        _sheet = img
        resolve(img)
      }

      img.onerror = reject
      img.src = eggSheetUrl
    })
  }

  return _sheetLoading
}

interface PixelEggSpriteProps {
  mode: 'bounce' | 'hatch'
  /** On-screen size (px, square). */
  size: number
  /**
   * Slot position in a grid of eggs. Used to deterministically spread each egg's
   * first bounce across the rest window so neighbours never stir together (random
   * jitter alone can collide with only a handful of eggs).
   */
  index?: number
  className?: string
  style?: CSSProperties
  /** Fired once when a `hatch` run reaches the final frame. */
  onDone?: () => void
}

export function PixelEggSprite({ mode, size, index = 0, className, style, onDone }: PixelEggSpriteProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const onDoneRef = useRef(onDone)
  onDoneRef.current = onDone

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')

    if (!canvas || !ctx) {
      return
    }

    const dpr = Math.min(window.devicePixelRatio || 1, 3)
    const dim = Math.round(size * dpr)
    canvas.width = dim
    canvas.height = dim

    const lastFrame = TOTAL_FRAMES - 1
    // Mild per-egg speed jitter so bounces don't feel mechanical.
    const frameMs = (mode === 'bounce' ? BOUNCE_MS : HATCH_MS) * (0.85 + Math.random() * 0.3)
    const restMs = () => REST_MIN_MS + Math.random() * (REST_MAX_MS - REST_MIN_MS)
    // First bounce: a deterministic per-slot slice of the rest window (so two
    // eggs never start together) plus a little random jitter on top.
    const firstDelay = ((index % 4) + 1) * (REST_MIN_MS / 4) + Math.random() * REST_MIN_MS

    // 32×32 offscreen we recolor per frame, then scale up nearest-neighbor.
    const off = document.createElement('canvas')
    off.width = FRAME
    off.height = FRAME
    const offCtx = off.getContext('2d', { willReadFrequently: true })

    let sheet: HTMLImageElement | null = null
    void loadSheet().then(img => {
      sheet = img
    })

    const render = (frame: number) => {
      if (!sheet || !offCtx) {
        return
      }

      offCtx.clearRect(0, 0, FRAME, FRAME)
      offCtx.imageSmoothingEnabled = false
      offCtx.drawImage(sheet, 0, frame * FRAME, FRAME, FRAME, 0, 0, FRAME, FRAME)
      const img = offCtx.getImageData(0, 0, FRAME, FRAME)
      const d = img.data

      for (let i = 0; i < d.length; i += 4) {
        if (d[i + 3] === 0) {
          continue
        }

        const g = d[i] * 3
        d[i] = CREME_LUT[g]
        d[i + 1] = CREME_LUT[g + 1]
        d[i + 2] = CREME_LUT[g + 2]
      }

      offCtx.putImageData(img, 0, 0)

      ctx.clearRect(0, 0, dim, dim)
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(off, 0, 0, FRAME, FRAME, 0, 0, dim, dim)
    }

    let raf = 0
    let step = 0
    let finished = false
    // bounce: `nextAt` is when the next thing happens — the next bounce frame, or
    // the start of a new bounce after a rest. hatch: `lastHatch` time-gates frames.
    let resting = mode === 'bounce'
    let nextAt = 0
    let lastHatch = 0

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)

      if (!sheet) {
        return
      }

      if (mode === 'hatch') {
        if (!lastHatch) {
          lastHatch = now
          render(HATCH_START)

          return
        }

        if (now - lastHatch < frameMs) {
          return
        }

        lastHatch = now
        const frame = Math.min(HATCH_START + step, lastFrame)
        render(frame)

        if (frame >= lastFrame) {
          if (!finished) {
            finished = true
            onDoneRef.current?.()
          }

          return // hold the cracked-open last frame
        }

        step += 1

        return
      }

      // bounce: rest on frame 0, play 0..5, then rest again.
      if (!nextAt) {
        render(0)
        nextAt = now + firstDelay // staggered first bounce, per slot

        return
      }

      if (now < nextAt) {
        return
      }

      if (resting) {
        resting = false
        step = 0
        render(0)
        nextAt = now + frameMs

        return
      }

      step += 1

      if (step >= BOUNCE_FRAMES) {
        resting = true
        render(0)
        nextAt = now + restMs()

        return
      }

      render(step)
      nextAt = now + frameMs
    }

    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
    }
  }, [mode, size, index])

  return (
    <canvas
      className={className}
      ref={canvasRef}
      style={{ width: size, height: size, imageRendering: 'pixelated', ...style }}
    />
  )
}
