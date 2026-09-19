import type { ComponentProps } from 'react'
import { memo } from 'react'

import { cn } from '@/lib/utils'

export type StatusTone = 'good' | 'muted' | 'warn' | 'bad'

const TONE_BG: Record<StatusTone, string> = {
  good: 'bg-primary',
  muted: 'bg-muted-foreground/40',
  warn: 'bg-amber-500',
  bad: 'bg-destructive'
}

// A quiet, finite breath for the "good" tone — a soft opacity pulse that reads
// as "alive and healthy" without keeping a compositor animation active for a
// durable connected row. Gateway-menu can stack two or three `good` dots; a
// chorus of permanent pinging rings would clutter the compact panel.
//
// Reduced-motion: the blanket override in styles.css
// (animation-duration: 0.01ms !important) neutralizes the pulse for users who
// ask for stillness. The `good` tone still reads as "online" via its brighter
// `bg-primary` background — no motion fallback needed (cf. quest-glow L745,
// which needs a box-shadow fallback because its warning signal is the shadow).
// Matches the spirit of #47942: don't strip the signal with the animation.
// See `@keyframes status-dot-breath` in styles.css.
const BREATH_GOOD = 'status-dot-breath'
const BREATH_GOOD_STYLE = { animationIterationCount: 2 }

interface StatusDotProps extends ComponentProps<'span'> {
  tone: StatusTone
}

export const StatusDot = memo(function StatusDot({ className, style, tone, ...props }: StatusDotProps) {
  return (
    <span
      aria-hidden="true"
      className={cn('inline-block size-1.5 rounded-full', TONE_BG[tone], tone === 'good' && BREATH_GOOD, className)}
      {...props}
      style={tone === 'good' ? { ...style, ...BREATH_GOOD_STYLE } : style}
    />
  )
})
