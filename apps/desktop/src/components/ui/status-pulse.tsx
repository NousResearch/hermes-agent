import { type ComponentProps, useEffect, useRef } from 'react'

import { createRendererLoopPauseController } from '@/lib/renderer-loop-pause'

const PULSE_DURATION_MS = 400
const PULSE_PERIOD_MS = 5_000

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

  useEffect(() => {
    const element = ref.current

    if (
      !element ||
      typeof element.animate !== 'function' ||
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    ) {
      return
    }

    let animation: Animation | null = null
    let timer = 0
    let stopped = false
    let pauseController: ReturnType<typeof createRendererLoopPauseController> | null = null

    const clearScheduled = () => {
      if (timer !== 0) {
        window.clearTimeout(timer)
        timer = 0
      }

      animation?.cancel()
      animation = null
    }

    const play = () => {
      timer = 0

      if (stopped || pauseController?.isPaused()) {
        return
      }

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
      timer = window.setTimeout(play, PULSE_PERIOD_MS)
    }

    const handlePauseChange = () => {
      clearScheduled()

      if (!pauseController?.isPaused()) {
        play()
      }
    }

    pauseController = createRendererLoopPauseController(handlePauseChange)
    play()

    return () => {
      stopped = true
      clearScheduled()
      pauseController?.dispose()
    }
  }, [kind, opacity])

  return <span {...props} ref={ref} />
}
