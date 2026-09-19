import { type ComponentProps, useEffect, useState } from 'react'

import { prefersReducedMotion } from '@/hooks/use-media-query'
import { createRendererLoopPauseController } from '@/lib/renderer-loop-pause'
import { cn } from '@/lib/utils'

/**
 * DecodeText — the "CONNECTING" scramble-decode effect as a reusable
 * primitive (extracted from gateway-connecting-overlay.tsx; same mechanics):
 *
 *  - Even-weight mono ascii charset so cycling glyphs never jump width
 *    (matches the nousnet-web download-button decode effect).
 *  - Decode resolves half a character per 45ms tick; when fully resolved it
 *    holds for 16 ticks, then (in loop mode) replays.
 *  - The first `prefix` characters NEVER scramble — split at render level so
 *    no timer logic (even a stale HMR one) can garble them.
 *  - Optional blinking dither-cursor square.
 *
 * Typography (mono, small, uppercase, wide tracking) is baked in; color comes
 * from the caller via className/text color so the same primitive works on the
 * boot overlay (--theme-primary) and quiet surfaces (--ui-text-quaternary).
 */

export const DECODE_SCRAMBLE_CHARS = '/\\|-_=+<>~:*'
const TICK_MS = 45
const HOLD_TICKS = 16

function scrambled(tail: string, resolvedCount: number): string {
  return Array.from(tail, (ch, i) =>
    ch === ' ' || i < resolvedCount ? ch : DECODE_SCRAMBLE_CHARS[(Math.random() * DECODE_SCRAMBLE_CHARS.length) | 0]
  ).join('')
}

export interface DecodeTextProps extends Omit<ComponentProps<'span'>, 'prefix'> {
  text: string
  /** Leading character count that stays legible at all times. */
  prefix?: number
  /** Run the decode. When false, renders the plain resolved text (used to
   *  freeze the word during exit choreography). */
  active?: boolean
  /** Replay after the hold, or resolve once and stop. */
  loop?: boolean
  /** Blinking dither-cursor square after the text. */
  cursor?: boolean
}

export function DecodeText({
  active = true,
  className,
  cursor = false,
  loop = true,
  prefix = 0,
  text,
  ...props
}: DecodeTextProps) {
  const staticPrefix = text.slice(0, prefix)
  const tailText = text.slice(prefix)
  const [tail, setTail] = useState(tailText)

  useEffect(() => {
    if (!active) {
      setTail(tailText)

      return
    }

    // Under reduced motion, skip the scramble interval and render the fully
    // resolved text immediately. The cursor blink (CSS animation) is also
    // killed by the blanket reduced-motion CSS rule.
    if (prefersReducedMotion()) {
      setTail(tailText)

      return
    }

    let resolved = 0
    let hold = 0
    let id = 0
    let pauseController: ReturnType<typeof createRendererLoopPauseController> | null = null

    const tick = () => {
      if (resolved >= tailText.length) {
        // The resolved text is already on screen. A one-shot decode is done;
        // a looping one only counts hold ticks — no state update per tick.
        if (!loop) {
          stop()
        } else if (++hold > HOLD_TICKS) {
          resolved = 0
          hold = 0
        }

        return
      }

      resolved += 0.5
      setTail(resolved >= tailText.length ? tailText : scrambled(tailText, Math.floor(resolved)))
    }

    const stop = () => {
      if (id !== 0) {
        window.clearInterval(id)
        id = 0
      }
    }

    // A looping decode is a 22Hz React update loop. Like the other renderer
    // loops, park it while the window is hidden/minimized and resume on return.
    const sync = () => {
      if (pauseController?.isPaused()) {
        stop()
      } else if (id === 0 && (loop || resolved < tailText.length)) {
        id = window.setInterval(tick, TICK_MS)
      }
    }

    pauseController = createRendererLoopPauseController(sync)
    sync()

    return () => {
      stop()
      pauseController?.dispose()
    }
  }, [active, loop, tailText])

  return (
    <span
      className={cn(
        'inline-flex items-center font-mono text-[0.64rem] font-semibold uppercase tracking-[0.4em] tabular-nums',
        className
      )}
      {...props}
    >
      {cursor && <style>{'@keyframes decode-cursor { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }'}</style>}
      {staticPrefix}
      {tail}
      {cursor && (
        <span
          aria-hidden="true"
          className="dither ml-0.5 inline-block size-2 shrink-0 -translate-y-px rounded-[1px]"
          style={{ animation: 'decode-cursor 1s step-end infinite' }}
        />
      )}
    </span>
  )
}
