import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { useEnterAnimation } from './use-enter-animation'

interface PlayedAnimation {
  keyframes: Keyframe[]
  options: KeyframeAnimationOptions
}

/**
 * Mounts one element through the hook and reports the animation it played, if
 * any. jsdom has no Web Animations API, so `animate` is the seam.
 */
function mountAnimated(enabled: boolean, animationKey?: string): PlayedAnimation | undefined {
  let played: PlayedAnimation | undefined

  function Probe() {
    const ref = useEnterAnimation(enabled, animationKey)

    return <div ref={ref} />
  }

  // Defined rather than spied on: jsdom ships no Web Animations API at all, so
  // there is no `animate` to wrap.
  Object.defineProperty(HTMLElement.prototype, 'animate', {
    configurable: true,
    value: (keyframes: Keyframe[], options: KeyframeAnimationOptions) => {
      played = { keyframes, options }

      return {} as Animation
    },
    writable: true
  })
  render(<Probe />)

  return played
}

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(HTMLElement.prototype, 'animate')
})

describe('useEnterAnimation', () => {
  it('plays once on mount', () => {
    const played = mountAnimated(true, 'plays-once')

    expect(played).toBeDefined()
    expect(played?.keyframes[0]).toMatchObject({ opacity: 0 })
  })

  it('stays out of the way when disabled', () => {
    expect(mountAnimated(false, 'disabled')).toBeUndefined()
  })

  // A key is only banked once the node survives a microtask, so that a mount
  // React immediately tears down doesn't burn it.
  it('does not replay for a key that already animated', async () => {
    expect(mountAnimated(true, 'replay')).toBeDefined()
    await Promise.resolve()

    expect(mountAnimated(true, 'replay')).toBeUndefined()
  })

  /**
   * The animation fills forwards, so any value in its last keyframe is held in
   * the animation origin of the cascade for the life of the element — above
   * the stylesheet. Naming an end opacity therefore doesn't just finish the
   * fade, it permanently overrules whatever opacity CSS wants the element to
   * rest at, and transcript scaffolding rests dimmed. Rows that animated in
   * during the turn stayed bright while their rehydrated neighbours faded, and
   * no hover could lift the bright ones because the sheet had lost the
   * argument. Opacity has to be left to CSS at the end.
   */
  it('leaves the resting opacity to the stylesheet', () => {
    const played = mountAnimated(true, 'resting-opacity')

    expect(played?.options.fill).toBe('both')
    expect(played?.keyframes.at(-1)).not.toHaveProperty('opacity')
  })
})
