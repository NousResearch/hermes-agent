import robot from '@/assets/figma-agent-robot.svg'
import { cn } from '@/lib/utils'

interface FigmaAgentIconProps {
  color?: string | null
  className?: string
}

/** The exported sidebar robot; profile colors remain user-owned. */
export function FigmaAgentIcon({ color, className }: FigmaAgentIconProps) {
  return (
    <span
      aria-hidden
      className={cn('inline-flex size-5 shrink-0 overflow-hidden rounded-sm', className)}
      style={{ backgroundColor: color || 'var(--portal-brand-blue)' }}
    >
      <img alt="" className="size-full" src={robot} />
    </span>
  )
}
