import { describe, expect, it } from 'vitest'

import { HUD_FOLLOW_BAR_HEIGHT, HUD_FOLLOW_GAP, hudFollowStep, hudFollowTarget } from './hud-follow'

const WORK = { x: 0, y: 0, width: 1920, height: 1040 }
const SIZE = { width: 620, height: 320 }

const BAR = HUD_FOLLOW_BAR_HEIGHT

const base = {
  workArea: WORK,
  barHeight: BAR,
  zoomFactor: 1,
  gap: HUD_FOLLOW_GAP,
  reach: 40,
  deadZone: 12,
  ease: 0.5,
  lastTarget: null
}

describe('hudFollowTarget', () => {
  it('parks the bar just to the lower-right of the pointer', () => {
    expect(hudFollowTarget({ x: 900, y: 300 }, SIZE, BAR, 1, WORK, 18)).toEqual({ x: 918, y: 318 })
  })

  it('flips to the left of the pointer near the right edge', () => {
    expect(hudFollowTarget({ x: 1900, y: 300 }, SIZE, BAR, 1, WORK, 18)).toEqual({ x: 1900 - 18 - 620, y: 318 })
  })

  it('flips the BAR above the pointer near the bottom, ignoring the transparent band', () => {
    // 1000 + 18 + 130 > 1040 → flip; the window's full 320 px height is not what has to fit.
    expect(hudFollowTarget({ x: 900, y: 1000 }, SIZE, BAR, 1, WORK, 18)).toEqual({ x: 918, y: 1000 - 18 - 130 })
    // 700 + 18 + 320 would not fit as a window, but the bar does, so no flip.
    expect(hudFollowTarget({ x: 900, y: 700 }, SIZE, BAR, 1, WORK, 18)).toEqual({ x: 918, y: 718 })
  })

  it('clamps so a sliver always stays on screen', () => {
    const target = hudFollowTarget({ x: 5, y: 300 }, SIZE, BAR, 1, WORK, 18)

    expect(target.x).toBeGreaterThanOrEqual(WORK.x + 40 - SIZE.width)
  })

  it('scales the offset with page zoom', () => {
    const zoomed = hudFollowTarget({ x: 900, y: 300 }, SIZE, BAR, 1.5, WORK, 18)

    expect(zoomed).toEqual({ x: 927, y: 327 })
  })
})

describe('hudFollowStep', () => {
  it('holds still while the pointer is on or within reach of the BAR, not the band', () => {
    const bounds = { x: 600, y: 400, ...SIZE }

    // On the bar.
    expect(hudFollowStep({ ...base, cursor: { x: 700, y: 450 }, bounds })).toEqual({
      origin: null,
      target: null,
      hold: 'in-reach',
      position: null
    })
    // Resting where the bar parks it (18 px up-left of the corner), bar at rest: hold.
    const rested = { ...base, lastTarget: { x: 600, y: 400 } }
    expect(hudFollowStep({ ...rested, cursor: { x: 582, y: 382 }, bounds }).hold).toBe('in-reach')
    // Pointer coming for a resting bar: hold — the bar must not back away.
    expect(hudFollowStep({ ...rested, cursor: { x: 575, y: 375 }, bounds }).hold).toBe('in-reach')
    // In the transparent band below the bar: that is moving away, follow.
    expect(hudFollowStep({ ...rested, cursor: { x: 700, y: 400 + BAR + 60 }, bounds }).hold).toBeNull()
  })

  it('keeps travelling to its spot even when the pointer is already near', () => {
    // Bar ~19 px from where it should park beside this pointer, pointer within
    // reach of it, but no rest recorded: it keeps going.
    const bounds = { x: 600, y: 385, ...SIZE }
    const step = hudFollowStep({ ...base, cursor: { x: 570, y: 352 }, bounds })

    expect(step.hold).toBeNull()
    expect(step.target).toEqual({ x: 588, y: 370 })
    expect(step.origin).not.toBeNull()
  })

  it('eases toward the target and reports it', () => {
    const bounds = { x: 0, y: 0, ...SIZE }
    const step = hudFollowStep({ ...base, cursor: { x: 900, y: 300 }, bounds })

    expect(step.target).toEqual({ x: 918, y: 318 })
    expect(step.hold).toBeNull()
    // Half way there with ease 0.5.
    expect(step.origin).toEqual({ x: 459, y: 159 })
  })

  it('lands exactly on the target once it is within two pixels', () => {
    const bounds = { x: 917, y: 317, ...SIZE }
    const step = hudFollowStep({ ...base, cursor: { x: 900, y: 300 }, bounds })

    expect(step.origin).toEqual({ x: 918, y: 318 })
  })

  it('reports settled and does not move once on target', () => {
    const bounds = { x: 918, y: 318, ...SIZE }
    const step = hudFollowStep({ ...base, cursor: { x: 900, y: 300 }, bounds })

    expect(step).toEqual({ origin: null, target: { x: 918, y: 318 }, hold: 'settled', position: null })
  })

  it('ignores cursor wobble inside the dead zone', () => {
    const bounds = { x: 918, y: 318, ...SIZE }
    const lastTarget = { x: 918, y: 318 }
    const step = hudFollowStep({ ...base, lastTarget, cursor: { x: 895, y: 295 }, bounds })

    expect(step.target).toEqual(lastTarget)
    expect(step.origin).toBeNull()
  })

  it('treats a bogus ease as a full step', () => {
    const bounds = { x: 0, y: 0, ...SIZE }
    const step = hudFollowStep({ ...base, ease: Number.NaN, cursor: { x: 900, y: 300 }, bounds })

    expect(step.origin).toEqual({ x: 918, y: 318 })
  })

  it('eases from the fractional position it was handed, not the rounded bounds', () => {
    const bounds = { x: 100, y: 100, ...SIZE }
    const step = hudFollowStep({ ...base, ease: 0.1, cursor: { x: 900, y: 300 }, bounds, position: { x: 100.4, y: 100.4 } })

    expect(step.position).toEqual({ x: 100.4 + (918 - 100.4) * 0.1, y: 100.4 + (318 - 100.4) * 0.1 })
    expect(step.origin).toEqual({ x: Math.round(step.position!.x), y: Math.round(step.position!.y) })
  })
})
