import robot from '@/assets/figma-agent-robot.svg'
import { cn } from '@/lib/utils'

interface FigmaAgentIconProps {
  color?: string | null
  className?: string
  variant?: 'solid' | 'subtle'
}

/** The exported sidebar robot; profile colors remain user-owned. */
export function FigmaAgentIcon({ color, className, variant = 'solid' }: FigmaAgentIconProps) {
  const tileColor = color || 'var(--portal-brand-blue)'
  const subtle = variant === 'subtle'

  return (
    <span
      aria-hidden
      className={cn('inline-flex size-5 shrink-0 overflow-hidden rounded-sm', className)}
      style={{
        backgroundColor: subtle ? `color-mix(in srgb, ${tileColor} 25%, transparent)` : tileColor,
        boxShadow: subtle ? `inset 0 0 0 0.5px ${tileColor}` : undefined
      }}
    >
      <img alt="" className={cn('size-full', subtle && 'brightness-0 dark:brightness-100')} src={robot} />
    </span>
  )
}
