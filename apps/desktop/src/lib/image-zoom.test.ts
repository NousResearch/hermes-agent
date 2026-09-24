import { describe, expect, it } from 'vitest'

import {
  MAX_ZOOM,
  MIN_ZOOM,
  ZOOM_STEP,
  clampPan,
  clampZoom,
  isVectorSource,
  isTypingTarget,
  maxPanOffset,
  zoomAtPoint,
  zoomBy,
  zoomFromWheel
} from './image-zoom'

const BASE = { height: 400, width: 600 }

describe('clampZoom', () => {
  it('never goes below the fitted size', () => {
    // The lightbox opens fit-to-screen; zooming out past that would shrink the
    // image inside an already-fitted box.
    expect(clampZoom(0.2)).toBe(MIN_ZOOM)
  })

  it('caps zooming in', () => {
    expect(clampZoom(1000)).toBe(MAX_ZOOM)
  })

  it('survives a non-finite scale', () => {
    expect(clampZoom(Number.NaN)).toBe(MIN_ZOOM)
  })
})

describe('zoomFromWheel', () => {
  it('scrolling up zooms in and down zooms out', () => {
    expect(zoomFromWheel(2, -100)).toBeGreaterThan(2)
    expect(zoomFromWheel(2, 100)).toBeLessThan(2)
  })

  it('moves by a constant ratio, so repeated notches feel even', () => {
    // Exponential in the delta: the same scroll from 2x must scale the same way
    // it did from 1x, or zooming decelerates as you go deeper.
    const fromOne = zoomFromWheel(1, -100) / 1
    const fromTwo = zoomFromWheel(2, -100) / 2
    expect(fromTwo).toBeCloseTo(fromOne, 5)
  })

  it('ignores a zero delta', () => {
    expect(zoomFromWheel(1.5, 0)).toBe(1.5)
  })

  it('stays inside the bounds under a violent delta', () => {
    expect(zoomFromWheel(4, -100000)).toBe(MAX_ZOOM)
    expect(zoomFromWheel(4, 100000)).toBe(MIN_ZOOM)
  })
})

describe('maxPanOffset', () => {
  it('is zero at fit, so a fitted image cannot be dragged away', () => {
    expect(maxPanOffset(BASE.width, MIN_ZOOM)).toBe(0)
  })

  it('only the overflow is reachable', () => {
    // At 2x a 600px-wide image is 1200px: 300px hangs off each side.
    expect(maxPanOffset(600, 2)).toBe(300)
  })
})

describe('clampPan', () => {
  it('pins to center at fit', () => {
    const pinned = clampPan({ x: 999, y: -999 }, BASE, MIN_ZOOM)
    // toBe(0) over toEqual({x:0,y:0}): clamping to a zero bound can yield -0,
    // which is the same position but a different value to a deep-equal.
    expect(pinned.x).toBe(0)
    expect(Math.abs(pinned.y)).toBe(0)
  })

  it('never lets the image leave a gap at the edge', () => {
    const panned = clampPan({ x: 10_000, y: 10_000 }, BASE, 2)
    expect(panned.x).toBe(maxPanOffset(BASE.width, 2))
    expect(panned.y).toBe(maxPanOffset(BASE.height, 2))
  })
})

describe('zoomAtPoint', () => {
  it('keeps the point under the cursor fixed while zooming', () => {
    // The contract that makes a diagram readable: zoom lands where you pointed,
    // instead of always pulling toward the middle.
    const cursor = { x: 100, y: 50 }
    const { offset, scale } = zoomAtPoint(cursor, { x: 0, y: 0 }, 1, 2, { height: 4000, width: 6000 })
    // Where the cursor's content sits after the transform.
    expect(cursor.x * scale + offset.x).toBeCloseTo(cursor.x * 1 + 0, 5)
    expect(cursor.y * scale + offset.y).toBeCloseTo(cursor.y * 1 + 0, 5)
  })

  it('returns to center when zooming back out to fit', () => {
    const zoomedIn = zoomAtPoint({ x: 120, y: 80 }, { x: 0, y: 0 }, 1, 3, BASE)
    const backToFit = zoomAtPoint({ x: 120, y: 80 }, zoomedIn.offset, zoomedIn.scale, MIN_ZOOM, BASE)
    expect(backToFit.scale).toBe(MIN_ZOOM)
    expect(backToFit.offset).toEqual({ x: 0, y: 0 })
  })

  it('leaves the offset inside the bounds', () => {
    const { offset, scale } = zoomAtPoint({ x: 300, y: 200 }, { x: 0, y: 0 }, 1, 2, BASE)
    expect(Math.abs(offset.x)).toBeLessThanOrEqual(maxPanOffset(BASE.width, scale))
    expect(Math.abs(offset.y)).toBeLessThanOrEqual(maxPanOffset(BASE.height, scale))
  })
})

describe('isVectorSource', () => {
  it('recognizes an svg file path', () => {
    expect(isVectorSource('/tmp/architecture.svg')).toBe(true)
  })

  it('recognizes an svg data url — how the preview pane loads a file', () => {
    // The right rail reads files as data URLs, so extension sniffing alone
    // would miss every SVG it renders and blur them all.
    expect(isVectorSource('data:image/svg+xml;base64,PHN2ZyB4bWxucz0i')).toBe(true)
  })

  it('ignores a query string or fragment after the extension', () => {
    expect(isVectorSource('/diagram.svg?v=2')).toBe(true)
    expect(isVectorSource('/diagram.svg#layer1')).toBe(true)
  })

  it('is false for rasters, which must keep transform-scale zoom', () => {
    expect(isVectorSource('/photo.png')).toBe(false)
    expect(isVectorSource('data:image/png;base64,iVBORw0KGgo=')).toBe(false)
  })

  it('is not fooled by a path that merely mentions svg', () => {
    expect(isVectorSource('/svg-exports/chart.png')).toBe(false)
  })

  it('handles a missing source', () => {
    expect(isVectorSource(undefined)).toBe(false)
  })
})

describe('isTypingTarget', () => {
  it('is true for a textarea — the chat composer', () => {
    // The bug this pins: a global `+` binding zoomed the preview while the
    // user was typing a message, so the character never reached the composer.
    const el = document.createElement('textarea')
    expect(isTypingTarget(el)).toBe(true)
  })

  it('is true for an input and a contenteditable', () => {
    expect(isTypingTarget(document.createElement('input'))).toBe(true)
    const editable = document.createElement('div')
    editable.contentEditable = 'true'
    // jsdom does not derive isContentEditable from the attribute.
    Object.defineProperty(editable, 'isContentEditable', { value: true })
    expect(isTypingTarget(editable)).toBe(true)
  })

  it('is false for a plain element, where a shortcut is welcome', () => {
    expect(isTypingTarget(document.createElement('div'))).toBe(false)
  })

  it('handles a null target', () => {
    expect(isTypingTarget(null)).toBe(false)
  })
})

describe('zoomBy', () => {
  it('a step in and a step out returns to where it started', () => {
    const stepped = zoomBy(zoomBy(2, ZOOM_STEP), 1 / ZOOM_STEP)
    expect(stepped).toBeCloseTo(2, 5)
  })
})
