import { useEffect, useLayoutEffect, useRef } from 'react'

import { cn } from '@/lib/utils'

interface TerminalOutputProps {
  className?: string
  text: string
}

const NEAR_BOTTOM_PX = 24

/**
 * Tiny read-only terminal viewer: monospace, non-wrapping (long lines scroll
 * horizontally), vertical scroll past `max-h`. Jumps to the bottom on mount,
 * then tails — sticking to the bottom as `text` grows, but only when the user
 * is already near the bottom so scrolling up to read earlier output isn't
 * interrupted.
 *
 * Self-contained so any surface (status rows, tool calls, inspectors) can drop
 * in a stdout/stderr box without re-implementing the scroll logic.
 */
export function TerminalOutput({ className, text }: TerminalOutputProps) {
  const ref = useRef<HTMLDivElement>(null)

  // On open: jump straight to the latest output (no animation, before paint).
  useLayoutEffect(() => {
    const el = ref.current

    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [])

  // On growth: tail only when already pinned near the bottom.
  useEffect(() => {
    const el = ref.current

    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX) {
      el.scrollTop = el.scrollHeight
    }
  }, [text])

  return (
    <div className={cn('max-h-16 overflow-auto overscroll-contain', className)} ref={ref}>
      <pre className="w-max min-w-full font-mono text-[0.5625rem] leading-[0.85rem] whitespace-pre text-muted-foreground/70">
        {text}
      </pre>
    </div>
  )
}
