import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  advanceOverlayVisualCandidate,
  clampOverlayRoamBounds,
  nearestOverlayHopDestination,
  overlayDropAllowed,
  overlayDropDestinations,
  overlayGroundY,
  overlayHasSupport,
  overlayHopApexY,
  overlayHopDestinations,
  overlayHopEndpointLanding,
  overlayHopMaxVerticalTravel,
  overlayHopYAtProgress,
  overlayIdleAction,
  overlayLandingAlongPath,
  overlayLowerPosition,
  overlayMotionLandingLedges,
  overlayMotionProbeIsCurrent,
  overlayPlannedAction,
  overlayRoamLedges,
  overlaySupportAt,
  overlaySupportMissOutcome,
  overlayVerticalCorrection,
  randomOverlayDropDestination,
  revalidateOverlayPlannedHop,
  usePetOverlayRoam
} from './use-pet-overlay-roam'

const originalDesktop = window.hermesDesktop

afterEach(() => {
  vi.restoreAllMocks()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
})

describe('overlay roam geometry', () => {
  it('keeps the visible pet inside a negative-coordinate display', () => {
    const area = { height: 1040, width: 1920, x: -1920, y: 0 }

    expect(clampOverlayRoamBounds(area, { height: 300, width: 240, x: -2300, y: 900 }, 64, 69)).toEqual({
      height: 300,
      width: 240,
      x: -2008,
      y: 768
    })
    expect(clampOverlayRoamBounds(area, { height: 300, width: 240, x: 200, y: -500 }, 64, 69)).toEqual({
      height: 300,
      width: 240,
      x: -152,
      y: -207
    })
  })

  it('places the overlay so its visible feet meet the usable display floor', () => {
    expect(overlayGroundY({ height: 1040, width: 1920, x: 0, y: 24 }, 300)).toBe(792)
  })

  it('accepts a nearby ledge as support but not a clearly lower one', () => {
    const bounds = { height: 300, width: 240, x: 200, y: 124 }

    expect(overlayHasSupport([{ left: 100, right: 400, y: 400 }], bounds)).toBe(true)
    expect(overlayHasSupport([{ left: 100, right: 400, y: 412 }], bounds)).toBe(false)
    expect(overlaySupportAt([{ left: 100, right: 400, y: 420 }], bounds, 24)).not.toBeNull()
  })

  it('uses a 35% hop, 15% drop, and 50% walk distribution on elevated supports', () => {
    expect(overlayIdleAction(true, () => 0.34)).toBe('hop')
    expect(overlayIdleAction(true, () => 0.35)).toBe('drop')
    expect(overlayIdleAction(true, () => 0.49)).toBe('drop')
    expect(overlayIdleAction(true, () => 0.5)).toBe('walk')
    expect(overlayIdleAction(true, () => 0.9)).toBe('walk')
  })

  it('moves the drop band to hops on the desktop floor', () => {
    expect(overlayIdleAction(false, () => 0.49)).toBe('hop')
    expect(overlayIdleAction(false, () => 0.5)).toBe('walk')
    expect(overlayIdleAction(false, () => 0.9)).toBe('walk')
  })

  it('keeps walking at 50% while preferring hops on lower supports', () => {
    expect(overlayIdleAction(true, () => 0.24, 0)).toBe('hop')
    expect(overlayIdleAction(true, () => 0.25, 0)).toBe('drop')
    expect(overlayIdleAction(true, () => 0.47, 0.9)).toBe('hop')
    expect(overlayIdleAction(true, () => 0.48, 0.9)).toBe('drop')
    expect(overlayIdleAction(true, () => 0.5, 0.9)).toBe('walk')
  })

  it('normalizes support height within positive and negative-coordinate work areas', () => {
    expect(overlayLowerPosition(100, { height: 800, width: 1200, x: 0, y: 100 })).toBe(0)
    expect(overlayLowerPosition(500, { height: 800, width: 1200, x: 0, y: 100 })).toBe(0.5)
    expect(overlayLowerPosition(900, { height: 800, width: 1200, x: 0, y: 100 })).toBe(1)
    expect(overlayLowerPosition(-500, { height: 800, width: 1200, x: -1200, y: -900 })).toBe(0.5)
  })

  it('targets only an upper support directly above, or hops in place', () => {
    const floor = { left: 0, right: 1000, y: 800 }
    const elevated = { left: 100, right: 500, y: 600 }
    const side = { left: 700, right: 900, y: 650 }

    expect(overlayHopDestinations([floor], floor, 400, 70)).toEqual([{ ledge: floor }])
    expect(overlayHopDestinations([floor, elevated], elevated, 300, 70)).toEqual([{ ledge: elevated }])
    expect(overlayHopDestinations([floor, side], floor, 300, 70)).toEqual([{ ledge: floor }])
    expect(overlayHopDestinations([floor, elevated, side], floor, 300, 70)).toEqual([{ ledge: elevated }])
  })

  it('forbids intentional drops when no hop destination exists', () => {
    expect(overlayDropAllowed(true, true)).toBe(true)
    expect(overlayDropAllowed(true, false)).toBe(false)
    expect(overlayDropAllowed(false, true)).toBe(false)
  })

  it('forces a grounded action after a drag or vertical motion to walk', () => {
    expect(overlayPlannedAction(true, 0, true, () => 0.1)).toBe('walk')
    expect(overlayPlannedAction(true, 0, true, () => 0.5)).toBe('walk')
    expect(overlayPlannedAction(true, 0, false, () => 0.1)).toBe('hop')
  })

  it('drops after three consecutive missed support probes', () => {
    const first = overlaySupportMissOutcome(0)
    const second = overlaySupportMissOutcome(first.failures)
    const third = overlaySupportMissOutcome(second.failures)

    expect(first).toEqual({ failures: 1, shouldDrop: false })
    expect(second).toEqual({ failures: 2, shouldDrop: false })
    expect(third).toEqual({ failures: 3, shouldDrop: true })
  })

  it('snaps small vertical edge jitter instead of starting a zero-distance motion', () => {
    expect(overlayVerticalCorrection(192, 200)).toBe('snap')
    expect(overlayVerticalCorrection(208, 200)).toBe('snap')
    expect(overlayVerticalCorrection(191, 200)).toBe('fall')
    expect(overlayVerticalCorrection(209, 200)).toBe('hop')
  })

  it('lands on the first valid surface crossed during descent', () => {
    const ledges = [
      { left: 0, right: 1000, y: 800 },
      { left: 150, right: 300, y: 400 },
      { left: 150, right: 300, y: 500 }
    ]

    expect(overlayLandingAlongPath(ledges, { x: 140, y: 60 }, { x: 220, y: 250 }, 300)).toEqual(ledges[1])
    expect(overlayLandingAlongPath(ledges, { x: 220, y: 250 }, { x: 100, y: 60 }, 300)).toBeNull()
    expect(overlayLandingAlongPath(ledges, { x: 0, y: 60 }, { x: 80, y: 250 }, 300)).toBeNull()
  })

  it('requires two consecutive visual samples before an airborne landing', () => {
    const ledge = { left: 100, right: 400, y: 400 }
    const first = advanceOverlayVisualCandidate(null, ledge)
    const second = advanceOverlayVisualCandidate(first, { ...ledge, y: 408 })

    expect(first).toEqual({ hits: 1, ledge })
    expect(second).toEqual({ hits: 2, ledge: { ...ledge, y: 408 } })
    expect(advanceOverlayVisualCandidate(second, null)).toBeNull()
  })

  it('keeps a planned visual destination eligible until the hop lands', () => {
    const floor = { left: 0, right: 1000, y: 800 }
    const destination = { left: 150, right: 300, y: 400 }
    const landingLedges = overlayMotionLandingLedges([floor], null, destination)

    expect(landingLedges).toEqual([floor, destination])
    expect(overlayLandingAlongPath(landingLedges, { x: 140, y: 60 }, { x: 220, y: 130 }, 300)).toEqual(destination)
    expect(overlayMotionLandingLedges([floor], destination, destination)).toEqual([floor, destination])
  })

  it('settles on the planned endpoint even when a frame skips the descending crossing', () => {
    const destination = { left: 150, right: 300, y: 400 }

    expect(overlayHopEndpointLanding(destination, 799, 800)).toBeNull()
    expect(overlayHopEndpointLanding(destination, 800, 800)).toBe(destination)
  })

  it('drops a planned hop destination after two refreshed scenes no longer contain it', () => {
    const destination = { left: 150, right: 300, y: 400 }
    const firstMiss = revalidateOverlayPlannedHop(destination, [], 0)
    const secondMiss = revalidateOverlayPlannedHop(firstMiss.ledge, [], firstMiss.failures)

    expect(firstMiss).toEqual({ failures: 1, ledge: destination })
    expect(secondMiss).toEqual({ failures: 2, ledge: null })
    expect(revalidateOverlayPlannedHop(destination, [{ ...destination, y: 406 }], 1)).toEqual({
      failures: 0,
      ledge: { ...destination, y: 406 }
    })
  })

  it('rejects surface probe results from a previous movement phase', () => {
    expect(overlayMotionProbeIsCurrent(4, 4, 'hop', 'hop')).toBe(true)
    expect(overlayMotionProbeIsCurrent(4, 5, 'hop', 'hop')).toBe(false)
    expect(overlayMotionProbeIsCurrent(4, 4, 'hop', 'walk')).toBe(false)
  })

  it('turns visible app-window top edges into ledges and removes covered spans', () => {
    const ledges = overlayRoamLedges(
      {
        windows: [
          { height: 300, width: 300, x: 100, y: 200 },
          { height: 300, width: 500, x: 50, y: 350 }
        ],
        workArea: { height: 800, width: 1200, x: 0, y: 0 }
      },
      240,
      64,
      70
    )

    expect(ledges).toEqual([
      { left: -88, right: 1048, y: 800 },
      { left: 12, right: 248, y: 200 },
      { left: 312, right: 398, y: 350 }
    ])
  })

  it('adds a visually detected horizontal surface without duplicating a native edge', () => {
    const ledges = overlayRoamLedges(
      {
        visualLedges: [
          { left: 100, right: 500, y: 310 },
          { left: 100, right: 500, y: 408 }
        ],
        windows: [{ height: 300, width: 500, x: 50, y: 400 }],
        workArea: { height: 800, width: 1200, x: 0, y: 0 }
      },
      240,
      64,
      70
    )

    expect(ledges).toEqual([
      { left: -88, right: 1048, y: 800 },
      { left: -38, right: 398, y: 400 },
      { left: 12, right: 348, y: 310 }
    ])
  })

  it('falls back to the display floor when a maximized front window covers everything', () => {
    const ledges = overlayRoamLedges(
      {
        windows: [
          { height: 800, width: 1200, x: 0, y: 0 },
          { height: 300, width: 500, x: 100, y: 300 }
        ],
        workArea: { height: 800, width: 1200, x: 0, y: 0 }
      },
      240,
      64,
      70
    )

    expect(ledges).toEqual([{ left: -88, right: 1048, y: 800 }])
  })

  it('limits upper support search to three rendered pet heights', () => {
    expect(overlayHopMaxVerticalTravel(70)).toBe(210)
  })

  it('passes through a destination-aware apex and lands at the target height', () => {
    const apex = overlayHopApexY(70, 420)

    expect(apex).toBe(385)
    expect(overlayHopYAtProgress(600, apex, 420, 0)).toBe(600)
    expect(overlayHopYAtProgress(600, apex, 420, 0.5)).toBe(385)
    expect(overlayHopYAtProgress(600, apex, 420, 1)).toBe(420)
  })

  it('chooses the nearest valid support as the hop destination', () => {
    const nearby = { ledge: { left: 150, right: 260, y: 500 } }
    const farther = { ledge: { left: 20, right: 180, y: 300 } }

    expect(nearestOverlayHopDestination([farther, nearby], 600)).toBe(nearby)
    expect(nearestOverlayHopDestination([], 600)).toBeNull()
  })

  it('limits upward candidates to three pet heights but keeps every lower candidate', () => {
    const current = { left: 100, right: 260, y: 400 }
    const nearbyUpper = { left: 100, right: 260, y: 250 }
    const tooHigh = { left: 100, right: 260, y: 100 }
    const lower = { left: 100, right: 260, y: 520 }
    const floor = { left: 0, right: 1000, y: 800 }
    const offColumn = { left: 400, right: 600, y: 600 }
    const ledges = [floor, current, nearbyUpper, tooHigh, lower, offColumn]

    expect(overlayHopDestinations(ledges, current, 180, 70)).toEqual([{ ledge: nearbyUpper }])
    expect(overlayDropDestinations(ledges, current, 180)).toEqual([{ ledge: floor }, { ledge: lower }])
  })

  it('selects a random pre-scanned drop destination', () => {
    const candidates = [
      { ledge: { left: 100, right: 260, y: 520 } },
      { ledge: { left: 0, right: 1000, y: 800 } }
    ]

    expect(randomOverlayDropDestination(candidates, () => 0)).toBe(candidates[0])
    expect(randomOverlayDropDestination(candidates, () => 0.999)).toBe(candidates[1])
    expect(randomOverlayDropDestination([], () => 0.5)).toBeNull()
  })

  it('replans immediately when the drag completion key changes', async () => {
    const roamEnvironment = vi.fn(async () => ({
      windows: [],
      workArea: { height: 800, width: 1200, x: 0, y: 0 }
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        petOverlay: {
          control: vi.fn(),
          roamEnvironment,
          setBounds: vi.fn()
        }
      } as unknown as Window['hermesDesktop']
    })
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(1)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})

    const isInteracting = () => false

    const { rerender, unmount } = renderHook(
      ({ replanKey }) =>
        usePetOverlayRoam({ enabled: true, isInteracting, loopMs: 1100, petH: 70, petW: 64, replanKey }),
      { initialProps: { replanKey: 0 } }
    )

    await waitFor(() => expect(roamEnvironment).toHaveBeenCalled())
    const callsBeforeRelease = roamEnvironment.mock.calls.length

    rerender({ replanKey: 1 })
    await waitFor(() => expect(roamEnvironment.mock.calls.length).toBeGreaterThan(callsBeforeRelease))

    unmount()
  })

  it('does not capture or analyze the desktop while a movement frame is running', async () => {
    const roamEnvironment = vi.fn(async () => ({
      windows: [],
      workArea: { height: 800, width: 1200, x: 0, y: 0 }
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        petOverlay: {
          control: vi.fn(),
          roamEnvironment,
          setBounds: vi.fn()
        }
      } as unknown as Window['hermesDesktop']
    })

    let nextFrame: FrameRequestCallback | undefined

    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      nextFrame = callback

      return 1
    })
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})

    const { unmount } = renderHook(() =>
      usePetOverlayRoam({ enabled: true, isInteracting: () => false, loopMs: 1100, petH: 70, petW: 64 })
    )

    await waitFor(() => expect(nextFrame).toBeDefined())
    const callsBeforeMovement = roamEnvironment.mock.calls.length
    const frame = nextFrame!

    act(() => frame(performance.now() + 20))

    expect(roamEnvironment).toHaveBeenCalledTimes(callsBeforeMovement)
    unmount()
  })
})
