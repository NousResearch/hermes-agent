/**
 * Inline ghost-text renderer.
 *
 * Lays a faded slash command on top of the composer's text layer at the
 * current caret position. The overlay is non-interactive (pointer-events:
 * none) so it never intercepts typing or focus — it is purely visual until
 * the user accepts the ghost with Tab / Shift+Tab.
 *
 * Hover detection is document-level: a `mousemove` listener checks whether
 * the pointer is inside the ghost command's rect. This keeps the overlay
 * fully pointer-transparent (no hot-zone that would eat clicks aimed at the
 * editor) while still showing the full skill description as a marquee strip
 * above the ghost: long descriptions scroll left in a seamless loop
 * (字幕跑马灯), short ones stay static.
 */
import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

export interface GhostSuggestionViewProps {
  /** Slash command to render in the ghost layer. Null hides the overlay. */
  command: string | null
  /** Full one-line description shown in the hover marquee. */
  description?: string | null
  /** Extra class on the floating layer — used by the editor to position it. */
  className?: string
}

const MARQUEE_SCROLL_MS = 6_000
const STATIC_MAX_CHARS = 14

/**
 * Measure the ghost width so the editor's text wrap stays consistent.
 * The composer reads the editor's first-line width and pins the ghost
 * to its right edge.
 */
function useGhostWidth(command: string | null): number {
  const [width, setWidth] = useState(0)

  useEffect(() => {
    if (!command) {
      setWidth(0)

      return
    }

    const measure = document.createElement('span')
    measure.textContent = ` ${command}`
    measure.style.position = 'absolute'
    measure.style.visibility = 'hidden'
    measure.style.whiteSpace = 'pre'
    measure.style.font = getComputedStyle(document.body).font
    document.body.appendChild(measure)
    setWidth(measure.getBoundingClientRect().width)
    measure.remove()
  }, [command])

  return width
}

export function GhostSuggestionView({ command, description, className }: GhostSuggestionViewProps) {
  const { t } = useI18n()
  const width = useGhostWidth(command)
  const [hovered, setHovered] = useState(false)
  const commandRef = useRef<HTMLSpanElement | null>(null)
  const trackRef = useRef<HTMLDivElement | null>(null)
  const [trackWidth, setTrackWidth] = useState(0)

  // Document-level hover detection: the overlay itself is pointer-events-none
  // (so clicks pass through to the editor), which means React's onMouseEnter
  // would never fire. Instead we watch mousemove globally and test the
  // pointer against the ghost command's live rect.
  useEffect(() => {
    if (!command) {
      setHovered(false)

      return
    }

    const onMove = (event: MouseEvent): void => {
      const el = commandRef.current

      if (!el) {
        setHovered(false)

        return
      }

      const rect = el.getBoundingClientRect()

      const inside =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom

      setHovered(inside)
    }

    document.addEventListener('mousemove', onMove)

    return () => document.removeEventListener('mousemove', onMove)
  }, [command])

  const showMarquee = Boolean(hovered && description && description.length > STATIC_MAX_CHARS)

  // Measure the marquee track so the animation duration scales with content
  // length (longer text scrolls at the same comfortable speed).
  useEffect(() => {
    if (!showMarquee) {
      setTrackWidth(0)

      return
    }

    const el = trackRef.current

    if (el) {
      setTrackWidth(el.scrollWidth)
    }
  }, [showMarquee, description])

  if (!command) {
    return null
  }

  const duration = Math.max(MARQUEE_SCROLL_MS, (trackWidth || 120) * 0.02)

  return (
    <div
      aria-hidden
      className={cn(
        'pointer-events-none absolute bottom-1 right-3 flex items-center gap-2',
        'text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)',
        className
      )}
      data-slot="composer-ghost-overlay"
    >
      {hovered && description ? (
        <div className="absolute bottom-full right-0 mb-1 max-w-[15rem] overflow-hidden rounded border border-(--ui-border-subtle) bg-(--ui-surface-secondary) px-2 py-1">
          {showMarquee ? (
            <div className="w-fit" ref={trackRef} style={{ whiteSpace: 'nowrap' }}>
              <div
                className="inline-flex"
                style={{
                  animation: `composer-ghost-marquee ${duration}ms linear infinite`,
                  whiteSpace: 'nowrap'
                }}
              >
                <span className="pr-6">{description}</span>
                <span aria-hidden className="pr-6">
                  {description}
                </span>
              </div>
            </div>
          ) : (
            <div className="truncate">{description}</div>
          )}
        </div>
      ) : null}
      <span
        className="font-mono italic opacity-70"
        ref={commandRef}
        style={{ minWidth: width ? `${width}px` : undefined }}
      >
        {' '}
        {command}
      </span>
      <span className="text-[0.65rem] uppercase tracking-wider opacity-60">
        {t.composer.ghostShiftTabHint}
      </span>
    </div>
  )
}
