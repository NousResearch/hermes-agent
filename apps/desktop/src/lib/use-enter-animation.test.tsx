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
    expect(played?.keyframes[0]).toMatchObject({ transform: 'translateY(0.375rem)' })
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
   * Opacity on any keyframe, with fill that holds while the document timeline
   * is paused (alt-tab, HUD hide, an unfocused window), pins the node at that
   * value. Opening at 0 left subagent rows and thinking blocks invisible.
   * Closing at 1 left scaffolding too bright for hover to lift. CSS already
   * rests those surfaces at 0.67. The slide is transform-only, and fill is
   * backwards so the offset applies on the first frame then the effect
   * releases instead of holding a compositor layer for the life of the node.
   */
  it('never animates opacity, and does not hold a filled transform', () => {
    const played = mountAnimated(true, 'resting-opacity')

    expect(played?.options.fill).toBe('backwards')

    for (const frame of played?.keyframes ?? []) {
      expect(frame).not.toHaveProperty('opacity')
    }
  })
})
