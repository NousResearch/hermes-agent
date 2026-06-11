import { useEffect, useState } from 'react'
import spinners, { type BrailleSpinnerName } from 'unicode-animations'

import { cn } from '@/lib/utils'

interface NormalisedSpinner {
  frames: readonly string[]
  interval: number
}

// Some spinners ship multi-character frames. Pull the first cell so each
// frame fits in one monospace box — matches how the TUI uses them.
const FRAMES_BY_NAME: Record<BrailleSpinnerName, NormalisedSpinner> = (() => {
  const out = {} as Record<BrailleSpinnerName, NormalisedSpinner>

  for (const name of Object.keys(spinners) as BrailleSpinnerName[]) {
    const raw = spinners[name]

    out[name] = {
      frames: raw.frames.map(frame => [...frame][0] ?? '⠀'),
      interval: raw.interval
    }
  }

  return out
})()

interface BrailleSpinnerProps {
  ariaLabel?: string
  className?: string
  spinner?: BrailleSpinnerName
}

/**
 * One-char braille spinner driven by `unicode-animations`. Mirrors the
 * spinner used by the Ink TUI so the desktop and terminal experiences
 * read the same visually. Renders inside an `inline-flex` cell with
 * `leading-none` and `items-center` so it sits vertically centred inside
 * its parent's line-box (e.g. the 1.1rem disclosure row).
 */
export function BrailleSpinner({ ariaLabel = 'Loading', className, spinner = 'breathe' }: BrailleSpinnerProps) {
  const spin = FRAMES_BY_NAME[spinner] ?? FRAMES_BY_NAME.breathe!
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    setFrame(0)
    const id = window.setInterval(() => setFrame(f => (f + 1) % spin.frames.length), spin.interval)

    return () => window.clearInterval(id)
  }, [spin])

  return (
    <span
      aria-label={ariaLabel}
      className={cn('inline-flex items-center justify-center font-mono leading-none tabular-nums', className)}
      role="status"
    >
      {spin.frames[frame]}
    </span>
  )
}
