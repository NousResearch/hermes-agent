import { describe, expect, it } from 'vitest'

import { NEIGHBOUR_MIN_PX, resizeSessionsSeam, SESSIONS_MIN_PX } from './section-resize'

describe('resizeSessionsSeam', () => {
  const start = { neighbour: 150, sessions: 300 }

  it('moves the seam both ways: what one side gives up, the other gains', () => {
    for (const delta of [-60, -10, 10, 60]) {
      const next = resizeSessionsSeam(start, delta)

      expect(next.neighbour).toBe(start.neighbour + delta)
      expect(next.sessions).toBe(start.sessions - delta)
    }
  })

  it('stops where either side hits its floor, and the total is conserved', () => {
    const neighbourFloor = resizeSessionsSeam(start, -1000)
    const sessionsFloor = resizeSessionsSeam(start, 1000)

    expect(neighbourFloor.neighbour).toBe(NEIGHBOUR_MIN_PX)
    expect(sessionsFloor.sessions).toBe(SESSIONS_MIN_PX)

    for (const next of [neighbourFloor, sessionsFloor]) {
      expect(next.neighbour + next.sessions).toBe(start.neighbour + start.sessions)
    }
  })

  it('conserves the total exactly on fractional measurements, drag after drag', () => {
    // Zoom and card rows make real heights fractional. Rounding either side
    // overflows the list (up) or leaks a pixel per drag (down).
    let heights = { neighbour: 78.37, sessions: 315.51 }
    const total = heights.neighbour + heights.sessions

    for (const delta of [-46, 31, -12.5, 40, -7]) {
      heights = resizeSessionsSeam(heights, delta)

      expect(heights.neighbour + heights.sessions).toBeCloseTo(total, 9)
    }
  })

  it('leaves a neighbour exactly where it was when the drag cannot move it', () => {
    // Sessions already at its floor: the neighbour has no room to grow, and a
    // fractional measurement must not round it a sub-pixel the wrong way.
    const start = { neighbour: 474.31, sessions: SESSIONS_MIN_PX }

    expect(resizeSessionsSeam(start, 40).neighbour).toBe(start.neighbour)
    expect(resizeSessionsSeam(start, 0).neighbour).toBe(start.neighbour)
  })
})
