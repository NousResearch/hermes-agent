import { describe, expect, it } from 'vitest'

import {
  clampFloatingGeometry,
  edgeSnapPlacement,
  type PreviewSnapSlot,
  surfaceGeometryForPlacement,
  WIN11_SNAP_LAYOUTS
} from './preview-surface-layout'

const viewport = { bottomInset: 36, height: 900, topInset: 40, width: 1200 }

describe('preview surface geometry', () => {
  it.each<[PreviewSnapSlot, object]>([
    ['left-half', { height: 824, width: 600, x: 0, y: 40 }],
    ['right-half', { height: 824, width: 600, x: 600, y: 40 }],
    ['top-half', { height: 412, width: 1200, x: 0, y: 40 }],
    ['bottom-half', { height: 412, width: 1200, x: 0, y: 452 }],
    ['left-third', { height: 824, width: 400, x: 0, y: 40 }],
    ['center-third', { height: 824, width: 400, x: 400, y: 40 }],
    ['right-third', { height: 824, width: 400, x: 800, y: 40 }],
    ['left-two-thirds', { height: 824, width: 800, x: 0, y: 40 }],
    ['right-two-thirds', { height: 824, width: 800, x: 400, y: 40 }],
    ['top-left-quarter', { height: 412, width: 600, x: 0, y: 40 }],
    ['top-right-quarter', { height: 412, width: 600, x: 600, y: 40 }],
    ['bottom-left-quarter', { height: 412, width: 600, x: 0, y: 452 }],
    ['bottom-right-quarter', { height: 412, width: 600, x: 600, y: 452 }]
  ])('computes %s inside the usable viewport', (placement, expected) => {
    expect(surfaceGeometryForPlacement(placement, viewport)).toEqual(expected)
  })

  it('keeps floating geometry within the usable viewport', () => {
    expect(clampFloatingGeometry({ height: 1200, width: 1800, x: -50, y: -60 }, viewport)).toEqual({
      height: 824,
      width: 1200,
      x: 0,
      y: 40
    })
  })

  it('exposes halves, asymmetric thirds, thirds, and quarters in snap layouts', () => {
    expect(WIN11_SNAP_LAYOUTS.map(layout => layout.slots)).toEqual([
      ['left-half', 'right-half'],
      ['left-two-thirds', 'right-third'],
      ['left-third', 'center-third', 'right-third'],
      ['top-left-quarter', 'top-right-quarter', 'bottom-left-quarter', 'bottom-right-quarter']
    ])
  })
})

describe('preview surface edge snapping', () => {
  it.each([
    [5, 45, 'top-left-quarter'],
    [1195, 45, 'top-right-quarter'],
    [5, 860, 'bottom-left-quarter'],
    [1195, 860, 'bottom-right-quarter'],
    [600, 45, 'maximized'],
    [5, 450, 'left-half'],
    [1195, 450, 'right-half'],
    [600, 860, 'bottom-half'],
    [600, 450, null]
  ] as const)('chooses the expected slot at (%s, %s)', (x, y, expected) => {
    expect(edgeSnapPlacement({ x, y }, viewport, 24)).toBe(expected)
  })
})
