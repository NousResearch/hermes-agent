import { type ReactNode } from 'react'

import { cn } from '@/lib/utils'

interface StatusRowProps {
  children: ReactNode
  className?: string
  /** Leading glyph slot (spinner / status dot / selection circle). */
  leading?: ReactNode
  /** Makes the whole row activatable (adds `cursor-pointer` + keyboard a11y).
   *  Trailing-slot buttons should `stopPropagation` so they don't also fire it. */
  onActivate?: () => void
  /** Right-aligned actions. Revealed on row hover/focus unless `trailingVisible`. */
  trailing?: ReactNode
  trailingVisible?: boolean
}

/**
 * Shared row chrome for everything in the composer status stack — status items
 * (subagents, background) AND queued prompts. Fixed height, a leading glyph
 * slot, flexible content, and a trailing actions slot that reveals on hover.
 * Hover background matches the session sidebar. Consumers fill the three slots;
 * they never re-implement the row container.
 */
export function StatusRow({
  children,
  className,
  leading,
  onActivate,
  trailing,
  trailingVisible = false
}: StatusRowProps) {
  return (
    <div
      className={cn(
        'group/status-row flex min-h-6 items-center gap-2 rounded-md px-1.5 py-1 hover:bg-(--ui-row-hover-background)',
        onActivate && 'cursor-pointer',
        className
      )}
      onClick={onActivate}
      onKeyDown={
        onActivate
          ? event => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onActivate()
              }
            }
          : undefined
      }
      role={onActivate ? 'button' : undefined}
      tabIndex={onActivate ? 0 : undefined}
    >
      <span className="flex size-3.5 shrink-0 items-center justify-center">{leading}</span>
      <div className="flex min-w-0 flex-1 items-center gap-2">{children}</div>
      {trailing && (
        <div
          className={cn(
            'flex shrink-0 items-center gap-0.5',
            !trailingVisible && 'opacity-0 group-hover/status-row:opacity-100 group-focus-within/status-row:opacity-100'
          )}
        >
          {trailing}
        </div>
      )}
    </div>
  )
}
