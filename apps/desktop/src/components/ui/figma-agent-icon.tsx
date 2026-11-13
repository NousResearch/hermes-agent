import robot from '@/assets/figma-agent-robot.svg'
import { cn } from '@/lib/utils'

interface FigmaAgentIconProps {
  color?: string | null
  className?: string
  /** Edge in px for a tile off the 20px default (a dialog row, a chat's empty state). */
  size?: number
  variant?: 'solid' | 'subtle' | 'outline'
}

/** Screenshot-backed agent marks; profile colors remain user-owned. */
export function FigmaAgentIcon({ color, className, size, variant = 'solid' }: FigmaAgentIconProps) {
  const tileColor = color || 'var(--portal-brand-fill)'
  const subtle = variant === 'subtle'
  const outline = variant === 'outline'

  return (
    <span
      aria-hidden
      className={cn('inline-flex size-5 shrink-0 overflow-hidden rounded-(--portal-radius-control,0.125rem)', className)}
      style={{
        backgroundColor:
          subtle || outline ? `color-mix(in srgb, ${tileColor} ${outline ? 12 : 25}%, transparent)` : tileColor,
        boxShadow: subtle ? `inset 0 0 0 0.5px ${tileColor}` : undefined,
        ...(size ? { borderRadius: size > 32 ? 'var(--portal-radius-surface)' : undefined, height: size, width: size } : {})
      }}
    >
      {outline ? (
        <svg
          aria-hidden
          className="m-auto size-[45%]"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <rect height="15" rx="3.5" width="17" x="3.5" y="7" />
          <path d="M12 3v4" />
        </svg>
      ) : (
        <img alt="" className={cn('size-full', subtle && 'brightness-0 dark:brightness-100')} src={robot} />
      )}
    </span>
  )
}
