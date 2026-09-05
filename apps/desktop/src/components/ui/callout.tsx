/**
 * A tone-washed advisory block: a left rule and a faint fill in some colour,
 * with a matching icon and title.
 *
 * NOT `Alert`. That one is a bordered, shadowed card with four fixed variants,
 * and it's what a page shows when it has one thing to say. This takes an
 * arbitrary tone, so it can carry a colour the surrounding data already
 * assigned — a diagnostic's severity, a lane's hue, a step's kind — and it
 * stacks quietly enough to appear several at a time inside a panel.
 */

import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

export function Callout({
  children,
  className,
  icon = 'warning',
  title,
  tone
}: {
  children?: ReactNode
  className?: string
  icon?: string
  title: ReactNode
  /** Any CSS colour — a token, a var, a literal. Both the rule and the wash
   *  are mixed from it, so one value tints the whole block. */
  tone: string
}) {
  return (
    <div
      className={cn('flex flex-col gap-2 rounded-md p-2.5', className)}
      style={{ backgroundColor: `color-mix(in srgb, ${tone} 7%, transparent)`, borderLeft: `2px solid ${tone}` }}
    >
      <div className="flex items-start gap-1.5 text-[0.75rem] font-medium" style={{ color: tone }}>
        <Codicon className="mt-px shrink-0" name={icon} size="0.8rem" />
        <span>{title}</span>
      </div>
      {children}
    </div>
  )
}
