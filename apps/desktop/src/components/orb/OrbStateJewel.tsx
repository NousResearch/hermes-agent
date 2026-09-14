import { type FC } from 'react'

import { cn } from '@/lib/utils'

import { OrbView, useOrbThinking } from './OrbView'
import { useOrbState } from './use-orb-state'

/**
 * Persistent assistant-state orb for the composer controls row.
 *
 * The thread's thinking dot only exists while a turn is visibly working; the
 * terminal and paused states — error, complete, idle, waiting for user — need
 * a home that is always mounted. This jewel is that home: it reads the same
 * `OrbState` the thread dot does, from the session stores alone (no message
 * scope), and stays decorative (`aria-hidden`) because the status rows
 * already narrate the state to assistive tech.
 *
 * Null when the user hasn't opted into the orb (Settings → Appearance), same
 * as the thread dot's StatusPulse fallback.
 */
export const OrbStateJewel: FC<{ className?: string }> = ({ className }) => {
  const { enabled, params } = useOrbThinking()
  const state = useOrbState()

  if (!enabled) {
    return null
  }

  return (
    <OrbView
      aria-hidden="true"
      className={cn('size-5 shrink-0', className)}
      fallbackType="original-thinking"
      params={params}
      state={state}
    />
  )
}
