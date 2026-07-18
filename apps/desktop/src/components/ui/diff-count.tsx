import { motion, useSpring, useTransform } from 'motion/react'
import { useEffect } from 'react'

import { cn } from '@/lib/utils'

// Snappy spring — fast transitions per the design.
const SPRING = { stiffness: 320, damping: 30, mass: 0.5 } as const

// A single integer that springs to its value via Motion (renders the motion
// value straight to the DOM, no per-frame React re-render). It initialises AT
// its value, so mounting/navigating shows it instantly — only a real change to
// the number (a live edit) springs it up/down. Switching threads in the same
// worktree (same numbers) therefore doesn't animate.
function AnimatedInt({ value }: { value: number }) {
  const spring = useSpring(value, SPRING)
  const text = useTransform(spring, latest => Math.round(latest).toString())

  useEffect(() => {
    spring.set(value)
  }, [value, spring])

  return <motion.span>{text}</motion.span>
}

interface DiffCountProps {
  added: number
  removed: number
  className?: string
}

/** Animated `+A −B` line-count, green/red via the top-level theme vars. Each
 *  number springs up/down via Motion (0 → value on first mount). */
export function DiffCount({ added, removed, className }: DiffCountProps) {
  if (!added && !removed) {
    return null
  }

  return (
    <span className={cn('flex shrink-0 items-center gap-1 tabular-nums', className)}>
      {added > 0 && (
        <span className="text-(--ui-green)">
          +<AnimatedInt value={added} />
        </span>
      )}
      {removed > 0 && (
        <span className="text-(--ui-red)">
          −<AnimatedInt value={removed} />
        </span>
      )}
    </span>
  )
}
