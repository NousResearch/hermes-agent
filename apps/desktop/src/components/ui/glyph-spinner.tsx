import './glyph-spinner.css'

import type { CSSProperties } from 'react'
import spinners, { type BrailleSpinnerName as SpinnerName } from 'unicode-animations'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { cn } from '@/lib/utils'

export type { SpinnerName }

interface NormalisedSpinner {
  frames: readonly string[]
  interval: number
}

/**
 * The two custom properties `glyph-spinner.css` reads off the strip to size
 * `steps()` and the cycle duration. Naming them in a type, rather than casting
 * the inline style with a bare `as CSSProperties`, makes a typo in a property
 * name a compile error instead of a silently dead declaration that falls back
 * to the braille defaults at runtime.
 */
export type GlyphSpinnerVars = CSSProperties & {
  '--glyph-spinner-duration': string
  '--glyph-spinner-frames': number
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
  /** Freeze the animation while the spinner stays mounted — for a caller that
   *  keeps it in the tree through a fade-out and does not want it animating
   *  once it is no longer the thing being waited on. */
  paused?: boolean
  spinner?: SpinnerName
}

/**
 * One-char glyph spinner driven by `unicode-animations` (braille, orbit, scan,
 * etc. — pick any `spinner` name). Mirrors the spinner used by the Ink TUI so
 * the desktop and terminal experiences read the same visually. Renders inside
 * an `inline-flex` cell with `leading-none` and `items-center` so it sits
 * vertically centred inside its parent's line-box.
 *
 * Animated entirely on the compositor — every frame is in the DOM from mount
 * and a transform keyframes animation scrolls between them. There is no timer
 * and no per-frame DOM write, because the ticker this replaced was scheduling
 * document-scale style recalculation on every tick (see glyph-spinner.css).
 *
 * The outer cell keeps the exact classes it always had, so consumer sizing
 * (`size-3`, `text-[0.75rem]`, colour, opacity) lands unchanged; the 1em-tall
 * clipping viewport is centred inside it by the same `items-center` that used
 * to centre the single glyph.
 */
export function GlyphSpinner({
  ariaLabel = 'Loading',
  className,
  paused = false,
  spinner = 'braille'
}: GlyphSpinnerProps) {
  const spin = FRAMES_BY_NAME[spinner] ?? FRAMES_BY_NAME.braille!
  // Pause when this surface is a hidden (kept-alive) tab: N mounted tabs each
  // animating burns CPU for pixels nobody can see. Window blur / minimize /
  // document-hidden are handled globally instead, by the
  // `:root[data-renderer-animations-paused]` rule in styles.css that
  // main.tsx's installRendererAnimationPauseState() drives — the same
  // mechanism every other continuous decorative animation already uses.
  // Note that only the primary window arms that attribute: a spinner mounted
  // in an overlay / quick / wake window keeps animating while its window is
  // blurred, exactly as the other decorative animations there do.
  const visible = usePaneVisible()

  const vars: GlyphSpinnerVars = {
    '--glyph-spinner-duration': `${spin.frames.length * spin.interval}ms`,
    '--glyph-spinner-frames': spin.frames.length
  }

  return (
    <span
      aria-label={ariaLabel}
      className={cn('inline-flex items-center justify-center font-mono leading-none tabular-nums', className)}
      role="status"
    >
      {/* Hidden from assistive tech: the accessible name is the label above.
          The frames are decorative, and this is a live region — the old
          implementation rewrote its text ~12x/second, which would announce a
          new glyph on every tick. */}
      <span aria-hidden="true" className="glyph-spinner" data-paused={paused || !visible ? 'true' : undefined}>
        <span className="glyph-spinner__strip" style={vars}>
          {spin.frames.map((frame, index) => (
            <span className="glyph-spinner__frame" key={`${index}:${frame}`}>
              {frame}
            </span>
          ))}
        </span>
      </span>
    </span>
  )
}
