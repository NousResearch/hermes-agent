import { useEffect, useState } from 'react'
import spinners, { type BrailleSpinnerName as SpinnerName } from 'unicode-animations'

import { cn } from '@/lib/utils'

export type { SpinnerName }

interface NormalisedSpinner {
  frames: readonly string[]
  interval: number
}

// Some spinners ship multi-character frames. Pull the first cell so each
// frame fits in one monospace box — matches how the TUI uses them.
const FRAMES_BY_NAME: Record<SpinnerName, NormalisedSpinner> = (() => {
  const out = {} as Record<SpinnerName, NormalisedSpinner>

  for (const name of Object.keys(spinners) as SpinnerName[]) {
    const raw = spinners[name]

    out[name] = {
      frames: raw.frames.map(frame => [...frame][0] ?? '⠀'),
      interval: raw.interval
    }
  }

  return out
})()

interface GlyphSpinnerProps {
  ariaLabel?: string
  className?: string
  spinner?: SpinnerName
}

/**
 * One-char glyph spinner driven by `unicode-animations` (braille, orbit, scan,
 * etc. — pick any `spinner` name). Mirrors the spinner used by the Ink TUI so
 * the desktop and terminal experiences read the same visually. Renders inside
 * an `inline-flex` cell with `leading-none` and `items-center` so it sits
 * vertically centred inside its parent's line-box.
 */
export function GlyphSpinner({ ariaLabel = 'Loading', className, spinner = 'braille' }: GlyphSpinnerProps) {
  const spin = FRAMES_BY_NAME[spinner] ?? FRAMES_BY_NAME.braille!
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
