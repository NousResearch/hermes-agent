import { expect, it } from 'vitest'

import { connectorPath, visiblePart } from './spotlight-geometry'

const box = (left: number, top: number, right: number, bottom: number) => ({ left, top, right, bottom })

it('points from the facing side of a row to the facing side of a nav spot on its left', () => {
  expect(connectorPath(box(460, 320, 980, 380), box(80, 500, 360, 650))).toBe('M 460 350 H 410 V 575 H 360')
})

it('mirrors for a right-to-left layout, where the nav sits on the right', () => {
  expect(connectorPath(box(100, 320, 600, 380), box(900, 40, 940, 70))).toBe('M 600 350 H 750 V 55 H 900')
})

it('draws nothing when the spot is not beside the row (stacked on a phone)', () => {
  expect(connectorPath(box(0, 300, 390, 340), box(10, 20, 50, 60))).toBeNull()
})

it('keeps only what every clip leaves on screen, and nothing when a clip hides it all', () => {
  const viewport = box(0, 0, 1280, 880)
  const scroller = box(60, 240, 380, 760)

  expect(visiblePart(box(80, 700, 360, 820), [viewport, scroller])).toEqual(box(80, 700, 360, 760))
  expect(visiblePart(box(80, 780, 360, 840), [viewport, scroller])).toBeNull()
})
