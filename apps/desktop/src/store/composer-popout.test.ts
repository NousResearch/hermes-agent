import { beforeEach, describe, expect, it } from 'vitest'

import {
  $composerPopoutZones,
  clampPopoutPosition,
  getComposerPopoutZone,
  POPOUT_WIDTH_REM,
  type PopoutBounds,
  pruneComposerPopoutZones,
  setComposerPopoutPosition,
  setComposerPoppedOut
} from './composer-popout'

// jsdom's window is 1024x768; every expectation below is relative to that.
const VW = 1024
const VH = 768

const BOX = { height: 56, width: 320 }

// A split layout: chat on the right half, under a 40px header.
const RIGHT_HALF: PopoutBounds = { bottom: VH, left: VW / 2, right: VW, top: 40 }
// The same drag intent, viewed from the left half.
const LEFT_HALF: PopoutBounds = { bottom: VH, left: 0, right: VW / 2, top: 40 }

const LEFT_ZONE = 'g-left'
const RIGHT_ZONE = 'g-right'

describe('clampPopoutPosition', () => {
  it('keeps the whole box inside its surface, not just its corner', () => {
    const { bottom, right } = clampPopoutPosition({ bottom: 24, right: 24 }, BOX, RIGHT_HALF)

    // Insets are viewport-relative, so convert back to an absolute rect.
    const boxRight = VW - right
    const boxLeft = boxRight - BOX.width
    const boxBottom = VH - bottom
    const boxTop = boxBottom - BOX.height

    expect(boxLeft).toBeGreaterThanOrEqual(RIGHT_HALF.left)
    expect(boxRight).toBeLessThanOrEqual(RIGHT_HALF.right)
    expect(boxTop).toBeGreaterThanOrEqual(RIGHT_HALF.top)
    expect(boxBottom).toBeLessThanOrEqual(RIGHT_HALF.bottom)
  })

  it('pulls an out-of-bounds intent back into the surface', () => {
    // Dragged far left of this surface (a position that belongs to the other
    // half of the split) — it must land inside, not off the edge.
    const { right } = clampPopoutPosition({ bottom: 24, right: 900 }, BOX, RIGHT_HALF)

    expect(VW - right - BOX.width).toBeGreaterThanOrEqual(RIGHT_HALF.left)
  })

  it('is pure — the same intent yields a different placement per surface', () => {
    const intent = { bottom: 24, right: 24 }

    const onRight = clampPopoutPosition(intent, BOX, RIGHT_HALF)
    const onLeft = clampPopoutPosition(intent, BOX, LEFT_HALF)

    // Right half honors the intent (it fits); left half has to push the box
    // inward. Two surfaces, two placements, one unchanged intent — this is what
    // lets keep-alive tabs share a zone's position without overwriting it.
    expect(onRight).not.toEqual(onLeft)
    expect(clampPopoutPosition(intent, BOX, RIGHT_HALF)).toEqual(onRight)
  })

  it('falls back to the full window when the surface has no measured area', () => {
    const { bottom, right } = clampPopoutPosition({ bottom: 24, right: 24 }, BOX, undefined)

    expect(bottom).toBe(24)
    expect(right).toBe(24)
  })

  it('assumes the compact float width when the box is unmeasured', () => {
    // Pre-layout (peel-off, restore) there is no rect yet; the fallback must
    // still leave the box grabbable rather than clamping it to a zero-width slot.
    const { right } = clampPopoutPosition({ bottom: 24, right: 5000 }, undefined, RIGHT_HALF)

    expect(VW - right - POPOUT_WIDTH_REM * 16).toBeGreaterThanOrEqual(RIGHT_HALF.left)
  })
})

describe('pop-out state is scoped to a layout zone', () => {
  beforeEach(() => {
    $composerPopoutZones.set({})
  })

  it('floats one zone without touching its neighbor', () => {
    setComposerPoppedOut(LEFT_ZONE, true)

    // Tabs in the left zone float; the right split stays docked. This is the
    // whole point of keying by zone: popping out on the left must not fling a
    // composer out of every other pane.
    expect(getComposerPopoutZone(LEFT_ZONE).poppedOut).toBe(true)
    expect(getComposerPopoutZone(RIGHT_ZONE).poppedOut).toBe(false)
  })

  it('keeps each zone position independent', () => {
    setComposerPopoutPosition(LEFT_ZONE, { bottom: 100, right: 100 })
    setComposerPopoutPosition(RIGHT_ZONE, { bottom: 24, right: 24 })

    expect(getComposerPopoutZone(LEFT_ZONE).position).toEqual({ bottom: 100, right: 100 })
    expect(getComposerPopoutZone(RIGHT_ZONE).position).toEqual({ bottom: 24, right: 24 })
  })

  it('reports the default for a zone nobody has touched', () => {
    // Every tab in a fresh zone reads the same default, so a new split starts
    // docked at the default corner rather than inheriting a neighbor's drag.
    expect(getComposerPopoutZone('g-unseen')).toEqual({ poppedOut: false, position: { bottom: 24, right: 24 } })
  })

  it('does not let one zone float seed the zones split after it', () => {
    setComposerPoppedOut(LEFT_ZONE, true)
    setComposerPopoutPosition(LEFT_ZONE, { bottom: 200, right: 200 })

    // A zone created later is its own thing — it must not inherit the float
    // state of whichever zone happened to be touched first.
    expect(getComposerPopoutZone('g-split-later')).toEqual({
      poppedOut: false,
      position: { bottom: 24, right: 24 }
    })
  })

  it('drops state for zones that no longer exist', () => {
    setComposerPoppedOut(LEFT_ZONE, true)
    setComposerPoppedOut(RIGHT_ZONE, true)

    pruneComposerPopoutZones([LEFT_ZONE])

    expect(Object.keys($composerPopoutZones.get())).toEqual([LEFT_ZONE])
  })
})
