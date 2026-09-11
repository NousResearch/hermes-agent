import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

/**
 * THE selection style — one vocabulary for every pickable thing in the shell
 * (chips, connector cards, layout cards): primary outline + tint when on, a
 * quiet neutral fill when off. No font-weight changes, no fills that shout.
 */
export const selectableClass = (on: boolean) =>
  cn(
    'border text-foreground transition-colors',
    on ? 'border-primary bg-primary/15' : 'border-transparent bg-muted hover:bg-accent/60'
  )

/** Toggleable chip — every pickable row/tag in the wizard. Two shapes:
 *  `card` (connector rows, roomier, fits an icon) and `pill` (compact
 *  tag-cloud toggles). */
export function Chip({
  className,
  icon,
  label,
  on,
  onToggle,
  sub,
  variant = 'card'
}: {
  className?: string
  icon?: ReactNode
  label: string
  on: boolean
  onToggle: () => void
  sub?: string
  variant?: 'card' | 'pill'
}) {
  return (
    <button
      aria-pressed={on}
      className={cn(
        'flex items-center text-left',
        variant === 'pill'
          ? 'max-w-full shrink-0 gap-1.5 rounded-full px-3 py-1.5 text-[12px] whitespace-normal wrap-anywhere'
          : 'min-w-0 gap-2.5 rounded-[6px] px-3 py-2.5 text-[13px]',
        selectableClass(on),
        className
      )}
      onClick={onToggle}
      type="button"
    >
      {icon}
      <span className="min-w-0">
        <span className={variant === 'pill' ? 'block wrap-anywhere' : 'block truncate'}>{label}</span>
        {sub && <span className="block text-xs text-muted-foreground">{sub}</span>}
      </span>
    </button>
  )
}
