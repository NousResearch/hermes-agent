import { describe, expect, it } from 'vitest'

import { type EnumeratedWindow, pickWindowBelow } from './window-below'

const win = (pid: number, x = 0, y = 0, width = 800, height = 600, app = `app-${pid}`): EnumeratedWindow => ({
  app,
  bounds: { x, y, width, height },
  id: pid * 10,
  pid,
  title: `${app} window`
})

const SELF_PID = 42
const SELF_BOUNDS = { x: 100, y: 100, width: 800, height: 600 }

describe('pickWindowBelow', () => {
  it('picks the first overlapping window behind ours in z-order', () => {
    const chrome = win(1, 120, 120)
    const spotify = win(2, 130, 130)

    const { below, frontmost } = pickWindowBelow([win(SELF_PID, 100, 100), chrome, spotify], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(chrome)
    expect(frontmost).toBe(chrome)
  })

  it('skips windows behind ours that do not overlap', () => {
    const elsewhere = win(1, 5000, 5000)
    const covered = win(2, 200, 200)

    const { below } = pickWindowBelow([win(SELF_PID, 100, 100), elsewhere, covered], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(covered)
  })

  it('skips our own other windows (same pid) while walking down', () => {
    const secondHermesWindow = win(SELF_PID, 150, 150)
    const target = win(7, 160, 160)

    const { below } = pickWindowBelow([win(SELF_PID, 100, 100), secondHermesWindow, target], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(target)
  })

  it('reports frontmost even when nothing overlaps', () => {
    const elsewhere = win(1, 5000, 5000)

    const { below, frontmost } = pickWindowBelow([win(SELF_PID, 100, 100), elsewhere], SELF_PID, SELF_BOUNDS)

    expect(below).toBeNull()
    expect(frontmost).toBe(elsewhere)
  })

  it('windows in front of ours are never "below", even overlapping', () => {
    const inFront = win(3, 110, 110)
    const behind = win(4, 120, 120)

    const { below, frontmost } = pickWindowBelow([inFront, win(SELF_PID, 100, 100), behind], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(behind)
    expect(frontmost).toBe(inFront)
  })

  it('falls back to overlap-only when our own window is not in the list', () => {
    // macOS omits windows the enumerator cannot see; still answer usefully.
    const chrome = win(1, 120, 120)
    const { below } = pickWindowBelow([chrome], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(chrome)
  })

  it('returns nulls for an empty enumeration', () => {
    const { below, frontmost } = pickWindowBelow([], SELF_PID, SELF_BOUNDS)

    expect(below).toBeNull()
    expect(frontmost).toBeNull()
  })

  it('edge-adjacent bounds do not count as overlap', () => {
    const adjacent = win(1, 900, 100) // starts exactly at our right edge

    const { below } = pickWindowBelow([win(SELF_PID, 100, 100), adjacent], SELF_PID, SELF_BOUNDS)

    expect(below).toBeNull()
  })
})
