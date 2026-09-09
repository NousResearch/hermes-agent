import type { CSSProperties } from 'react'

import { cn } from '@/lib/utils'

/**
 * The oversized display lettering of an empty chat — the Collapse face that
 * writes "HERMES AGENT" across a fresh draft, and a bot's name across its own
 * empty chat.
 *
 * The doubled children are load-bearing, not a typo: `.fit-text` sizes the
 * visible span from a container query and needs the `aria-hidden` twin laid
 * out beside it as the width reference. Anything that renders this lettering
 * comes through here rather than restating that structure.
 */
export function Wordmark({
  className,
  fitMin = '2.75rem',
  text,
  width = 'calc(100% - 1rem)'
}: {
  className?: string
  /** Floor for the fitted font size, as a CSS length. */
  fitMin?: string
  text: string
  /** How much of the column the lettering spans. `.fit-text` sizes to fill
   *  this, so a short word set at full width comes out enormous — a name gets
   *  less room than a twelve-character wordmark. */
  width?: string
}) {
  return (
    <p
      aria-label={text}
      className={cn(
        'wordmark fit-text mx-auto text-midground mix-blend-plus-lighter dark:text-foreground/90',
        className
      )}
      style={{ '--fit-min': fitMin, width } as CSSProperties}
    >
      <span>
        <span>{text}</span>
      </span>
      <span aria-hidden="true">{text}</span>
    </p>
  )
}
