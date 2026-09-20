import { describe, expect, it, vi } from 'vitest'

import { flipSurface } from './flip'

/** A surface that records the keyframes it is asked to animate. */
function surface() {
  const calls: Keyframe[][] = []

  const element = {
    animate: (frames: Keyframe[]) => {
      calls.push(frames)

      return { cancel: vi.fn(), finished: Promise.resolve() } as unknown as Animation
    }
  } as unknown as HTMLElement

  return { calls, element }
}

const blurs = (calls: Keyframe[][]) =>
  calls
    .flat()
    .map(frame => String(frame.filter ?? ''))
    .filter(Boolean)

describe('flipSurface', () => {
  it('softens the surface as it swings edge-on', async () => {
    const { calls, element } = surface()

    await flipSurface(element, () => {}, { direction: 1, reducedMotion: false, softwareComposited: false })

    // Flat facing the reader at both ends of the turn, at its softest in
    // between: the blur is what the movement does, not a state the page is in.
    expect(blurs(calls).at(0)).toBe('blur(0px)')
    expect(blurs(calls).at(-1)).toBe('blur(0px)')
    expect(blurs(calls).some(filter => filter !== 'blur(0px)')).toBe(true)
  })

  // Measured: on a GPU the blur costs about 1 ms of a 16.7 ms frame, and on the
  // CPU the same turn halves its frame count and reaches a 100 ms frame. The app
  // drops to CPU compositing by itself on a remote display, so this is the
  // difference between a turn and a slideshow for every SSH, VNC and RDP user.
  it('drops the blur, and only the blur, when the frames are drawn on the CPU', async () => {
    const soft = surface()
    const fast = surface()

    await flipSurface(soft.element, () => {}, { direction: 1, reducedMotion: false, softwareComposited: true })
    await flipSurface(fast.element, () => {}, { direction: 1, reducedMotion: false, softwareComposited: false })

    expect(blurs(soft.calls).every(filter => filter === 'blur(0px)')).toBe(true)

    const transforms = (calls: Keyframe[][]) => calls.flat().map(frame => frame.transform)

    expect(transforms(soft.calls)).toEqual(transforms(fast.calls))
  })

  it('asks for no turn at all when the reader has asked for less motion', async () => {
    const { calls, element } = surface()

    await flipSurface(element, () => {}, { direction: 1, reducedMotion: true, softwareComposited: false })

    expect(calls.flat().every(frame => frame.transform === undefined && frame.filter === undefined)).toBe(true)
    expect(calls.flat().map(frame => frame.opacity)).toEqual([1, 0, 0, 1])
  })

  it('swaps the sides between the two halves, whatever the motion looks like', async () => {
    const { element } = surface()
    const swap = vi.fn()

    await flipSurface(element, swap, { direction: -1, reducedMotion: true, softwareComposited: true })

    expect(swap).toHaveBeenCalledTimes(1)
  })
})
