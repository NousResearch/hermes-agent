import { describe, expect, it } from 'vitest'

import { clampChannelHoldSeconds } from './screen-annotations'
import {
  bandFractions,
  clampBandFraction,
  clampSampleHz,
  matchCapturerWindow,
  SUBTITLE_BAND_FRACTION_DEFAULT,
  SUBTITLE_SAMPLE_HZ_DEFAULT,
  subtitleBand,
  subtitleShapes
} from './subtitle-capture'

const DISPLAY = { height: 1000, width: 1600, x: 0, y: 0 }

describe('clamps', () => {
  it('sample rate: junk falls to the default, extremes clamp to the bounds', () => {
    expect(clampSampleHz(undefined)).toBe(SUBTITLE_SAMPLE_HZ_DEFAULT)
    expect(clampSampleHz(Number.NaN)).toBe(SUBTITLE_SAMPLE_HZ_DEFAULT)
    expect(clampSampleHz(0)).toBeGreaterThan(0)
    expect(clampSampleHz(100)).toBeLessThanOrEqual(8)
  })

  it('band fraction: junk falls to the default, extremes clamp to the bounds', () => {
    expect(clampBandFraction(undefined)).toBe(SUBTITLE_BAND_FRACTION_DEFAULT)
    expect(clampBandFraction(0.01)).toBeGreaterThanOrEqual(0.1)
    expect(clampBandFraction(0.9)).toBeLessThanOrEqual(0.5)
  })

  it('channel hold: junk falls to the default, a runaway hold clamps down', () => {
    const fallback = clampChannelHoldSeconds(undefined)

    expect(fallback).toBeGreaterThan(0)
    expect(clampChannelHoldSeconds(-3)).toBe(fallback)
    expect(clampChannelHoldSeconds(2)).toBe(2)
    expect(clampChannelHoldSeconds(10_000)).toBeLessThanOrEqual(60)
  })
})

describe('subtitleBand', () => {
  it('is the bottom fraction of the window', () => {
    const band = subtitleBand({ height: 800, width: 1200, x: 100, y: 50 }, DISPLAY, 0.25)

    expect(band).toEqual({ height: 200, width: 1200, x: 100, y: 650 })
  })

  it('clips to the display so a window hanging off-screen cannot produce crop coordinates outside the stream', () => {
    // Window hangs off the right edge and its band dips below the bottom edge:
    // raw band = {x:1000, y:900, w:1200, h:200} against a 1600x1000 display.
    const band = subtitleBand({ height: 800, width: 1200, x: 1000, y: 300 }, DISPLAY, 0.25)

    expect(band).toEqual({ height: 100, width: 600, x: 1000, y: 900 })
  })

  it('returns null for a window with no visible band area', () => {
    expect(subtitleBand({ height: 800, width: 1200, x: 5000, y: 50 }, DISPLAY, 0.25)).toBeNull()
  })

  it('round-trips through display fractions', () => {
    const band = subtitleBand({ height: 800, width: 1200, x: 100, y: 50 }, DISPLAY, 0.25)!
    const fractions = bandFractions(band, DISPLAY)

    expect(fractions.left * DISPLAY.width + DISPLAY.x).toBeCloseTo(band.x)
    expect(fractions.top * DISPLAY.height + DISPLAY.y).toBeCloseTo(band.y)
    expect(fractions.width * DISPLAY.width).toBeCloseTo(band.width)
    expect(fractions.height * DISPLAY.height).toBeCloseTo(band.height)
  })
})

describe('subtitleShapes', () => {
  const BAND = { height: 250, width: 1500, x: 10, y: 700 }
  const PARAMS = {
    band: BAND,
    box: { height: 60, width: 800, x: 240, y: 90 },
    cropHeight: 250,
    cropWidth: 1500,
    display: { height: 1000, width: 1600, x: 0, y: 0 },
    text: 'Vejo você do outro lado.'
  }

  it('paints an opaque steady cover over the original line and the translation centered on it', () => {
    const shapes = subtitleShapes(PARAMS)

    expect(shapes).toHaveLength(2)

    const [cover, label] = shapes

    expect(cover).toMatchObject({ color: 'black', fill: true, kind: 'rect', steady: true })
    expect(label).toMatchObject({ color: 'white', kind: 'label', steady: true, text: PARAMS.text })

    if (cover.kind !== 'rect' || label.kind !== 'label') {
      throw new Error('unreachable')
    }

    // Cover pads beyond the box so antialiased fringes of the original don't peek out.
    expect(cover.x).toBeLessThan(BAND.x + PARAMS.box.x)
    expect(cover.y).toBeLessThan(BAND.y + PARAMS.box.y)
    expect(cover.width).toBeGreaterThan(PARAMS.box.width)
    expect(cover.height).toBeGreaterThan(PARAMS.box.height)

    // Label anchors at the cover's horizontal center, baseline inside it.
    expect(label.x).toBe(Math.round(cover.x + cover.width / 2))
    expect(label.y).toBeGreaterThan(cover.y)
    expect(label.y).toBeLessThan(cover.y + cover.height)
  })

  it('maps crop pixels through the band scale (Retina-style 2x crop)', () => {
    const shapes = subtitleShapes({ ...PARAMS, cropHeight: 500, cropWidth: 3000 })
    const cover = shapes[0]

    if (cover.kind !== 'rect') {
      throw new Error('unreachable')
    }

    // Same box in a 2x-denser crop lands at half the DIP offset/size (plus pad).
    expect(cover.x).toBe(Math.round(BAND.x + PARAMS.box.x / 2) - 10)
    expect(cover.width).toBe(Math.round(PARAMS.box.width / 2) + 20)
  })

  it('derives the font from the original line height and caps it', () => {
    const shapes = subtitleShapes(PARAMS)
    const label = shapes[1]

    if (label.kind !== 'label') {
      throw new Error('unreachable')
    }

    expect(label.fontSize).toBeGreaterThanOrEqual(14)
    expect(label.fontSize).toBeLessThanOrEqual(64)

    const twoLines = subtitleShapes({ ...PARAMS, text: 'linha um\nlinha dois' })[1]

    if (twoLines.kind !== 'label') {
      throw new Error('unreachable')
    }

    // Two display lines share the same original box height → smaller font.
    expect(twoLines.fontSize!).toBeLessThan(label.fontSize!)
  })

  it('draws nothing for empty text or a degenerate box', () => {
    expect(subtitleShapes({ ...PARAMS, text: '   ' })).toEqual([])
    expect(subtitleShapes({ ...PARAMS, box: { height: 0, width: 800, x: 0, y: 0 } })).toEqual([])
    expect(subtitleShapes({ ...PARAMS, cropWidth: 0 })).toEqual([])
  })
})

describe('matchCapturerWindow', () => {
  const sources = [
    { id: 'window:99:0', name: 'Finder' },
    { id: 'window:42:0', name: 'Netflix - Google Chrome' }
  ]

  it('matches Chromium window:<id>: prefix first', () => {
    expect(matchCapturerWindow(sources, 42, { app: 'Chrome', title: 'Other' })?.id).toBe('window:42:0')
  })

  it('falls back to title substring when the id is unknown', () => {
    expect(matchCapturerWindow(sources, 7, { app: 'Safari', title: 'Netflix' })?.id).toBe('window:42:0')
  })
})
