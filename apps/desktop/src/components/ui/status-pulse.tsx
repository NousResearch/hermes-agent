import { useStore } from '@nanostores/react'
import { type ComponentProps, useEffect, useRef } from 'react'

import { createRendererLoopPauseController } from '@/lib/renderer-loop-pause'
import { $reducedEffects } from '@/store/reduced-effects'

const PULSE_DURATION_MS = 400
const PULSE_PERIOD_MS = 5_000

// One pause controller + one period timer shared by every StatusPulse
// instance. A sidebar can show dozens of pulsing dots at once; per-instance
// controllers would mean N×(document/window/bridge) listeners and N
// unsynchronized 5s wakes. Ref-counted: the controller and timer exist only
// while at least one pulse is mounted, and all pulses play in one aligned
// wake so the renderer sleeps between beats.
type PulseSubscriber = { play: () => void; cancel: () => void }

const pulseSubscribers = new Set<PulseSubscriber>()
let sharedPauseController: ReturnType<typeof createRendererLoopPauseController> | null = null
let sharedTimer = 0

const stopSharedTimer = () => {
  if (sharedTimer !== 0) {
    window.clearTimeout(sharedTimer)
    sharedTimer = 0
  }
}

const beat = () => {
  sharedTimer = 0

  if (sharedPauseController?.isPaused() || pulseSubscribers.size === 0) {
    return
  }

  for (const subscriber of pulseSubscribers) {
    subscriber.play()
  }

  sharedTimer = window.setTimeout(beat, PULSE_PERIOD_MS)
}

const handleSharedPauseChange = () => {
  stopSharedTimer()

  if (sharedPauseController?.isPaused()) {
    // Minimized/hidden: cancel in-flight animations so the compositor can
    // sleep immediately instead of finishing a pulse nobody sees.
    for (const subscriber of pulseSubscribers) {
      subscriber.cancel()
    }

    return
  }

  beat()
}

const subscribePulse = (subscriber: PulseSubscriber): (() => void) => {
  pulseSubscribers.add(subscriber)

  if (!sharedPauseController) {
    sharedPauseController = createRendererLoopPauseController(handleSharedPauseChange)
  }

  // First subscriber (or a new one joining mid-sleep): play immediately and
  // start the beat. Later joiners just wait for the next aligned beat.
  if (sharedTimer === 0 && !sharedPauseController.isPaused()) {
    subscriber.play()
    sharedTimer = window.setTimeout(beat, PULSE_PERIOD_MS)
  }

  return () => {
    pulseSubscribers.delete(subscriber)

    if (pulseSubscribers.size === 0) {
      stopSharedTimer()
      sharedPauseController?.dispose()
      sharedPauseController = null
    }
  }
}

export interface StatusPulseProps extends Omit<ComponentProps<'span'>, 'children' | 'ref'> {
  kind: 'opacity' | 'ping'
  opacity?: number
}

/**
 * A finite status pulse with a real sleep between plays.
 *
 * Continuous CSS animations keep Chromium producing frames and recalculating
 * styles for an otherwise motionless Desktop window. Drive the same visual
 * cue directly so React stays out of the loop and the renderer/compositor can
 * sleep between pulses.
 */
export function StatusPulse({ kind, opacity = 1, ...props }: StatusPulseProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const reducedEffects = useStore($reducedEffects)

  useEffect(() => {
    const element = ref.current

    if (
      !element ||
      typeof element.animate !== 'function' ||
      reducedEffects ||
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    ) {
      return
    }

    let animation: Animation | null = null

    const play = () => {
      animation?.cancel()
      animation = element.animate(
        kind === 'ping'
          ? [
              { opacity, transform: 'scale(1)' },
              { opacity: 0, transform: 'scale(2)' }
            ]
          : [{ opacity: 1 }, { opacity: 0.5 }, { opacity: 1 }],
        {
          duration: PULSE_DURATION_MS,
          easing: kind === 'ping' ? 'cubic-bezier(0, 0, 0.2, 1)' : 'ease-in-out',
          iterations: 1
        }
      )
    }

    const cancel = () => {
      animation?.cancel()
      animation = null
    }

    const unsubscribe = subscribePulse({ play, cancel })

    return () => {
      unsubscribe()
      cancel()
    }
  }, [kind, opacity, reducedEffects])

  return <span {...props} ref={ref} />
}
