import { useEffect, useRef, useState } from 'react'
import * as React from 'react'
import { createPortal } from 'react-dom'

import { cn } from '@/lib/utils'

import type { SessionInfo } from '@/types/hermes'

interface SessionHoverCardProps {
  /**
   * The session whose cached summary (or first-message preview) the
   * card surfaces. The card is silent when both `summary` and
   * `preview` are null/empty — no popover is mounted at all.
   */
  session: SessionInfo
  /**
   * The element that triggers the hover. Pass the row's existing
   * <button>; the card anchors to its bounding rect. The button
   * is rendered as-is — no clone, no wrapper, no portal inside
   * the row.
   */
  children: React.ReactElement
  /** Open delay in ms. Default 200 (matches the issue #45103 spec). */
  openDelay?: number
  /** Close delay in ms. Default 100 — shorter than open so a quick
   *  swipe across a row doesn't leave a stuck card. */
  closeDelay?: number
}

const NARROW_VIEWPORT_QUERY = '(max-width: 600px)'
const CARD_WIDTH_PX = 320
const VIEWPORT_EDGE_PADDING_PX = 8

/**
 * Hover card for a sidebar session row (issue #45103).
 *
 * Renders the cached AI summary (preferred) or the first-message
 * preview (fallback) in a small popover anchored to the trigger.
 * Pre-generated in the background by the SessionSummaryScheduler
 * (PR 3) — there is no LLM call on hover.
 *
 * No external Radix dep. Implemented with mouseenter/mouseleave +
 * setTimeout + a React portal, because @radix-ui/react-hover-card
 * is not in the project's main radix-ui bundle and adding a new
 * npm dependency for ~40 lines of behavior wasn't worth the
 * package-lock churn (AGENTS.md: "extend, don't duplicate" + the
 * pin-bound policy discourages add-on deps without strong
 * justification).
 *
 * Why a portal:
 *  - The trigger (sidebar row) lives inside an overflow-hidden
 *    scroll container. A non-portal'd popover would be clipped.
 *
 * Why a delay:
 *  - 200 ms open is the issue's spec. Stops the card from popping
 *    up while the user is sweeping the mouse across the list.
 *  - 100 ms close keeps the card responsive when the user moves
 *    the mouse to a target inside the card.
 *
 * Why the narrow viewport branch:
 *  - On mobile / narrow layouts the sidebar is collapsed by
 *    default and the hover gesture isn't a thing. The whole card
 *    is suppressed; the row itself shows a 2-sentence inline
 *    highlight via the `data-summary-mobile` attribute the row
 *    reads.
 */
export function SessionHoverCard({
  session,
  children,
  openDelay = 200,
  closeDelay = 100,
}: SessionHoverCardProps) {
  // The full body text — prefer the cached AI summary, fall back
  // to the first-message preview, otherwise null (no card at all).
  const body = session.summary?.trim() || session.preview?.trim() || null
  // On narrow viewports the whole card is suppressed; the row
  // itself gets a 2-sentence highlight via a data attribute
  // (rendered by the SidebarSessionRow in the v1 integration).
  const [narrow, setNarrow] = useState(false)
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return
    const mql = window.matchMedia(NARROW_VIEWPORT_QUERY)
    const update = () => setNarrow(mql.matches)
    update()
    mql.addEventListener?.('change', update)
    return () => mql.removeEventListener?.('change', update)
  }, [])

  if (!body || narrow) {
    return children
  }

  return (
    <SessionHoverCardTrigger
      body={body}
      closeDelay={closeDelay}
      openDelay={openDelay}
      session={session}
    >
      {children}
    </SessionHoverCardTrigger>
  )
}

function SessionHoverCardTrigger({
  session,
  body,
  openDelay,
  closeDelay,
  children,
}: Required<Pick<SessionHoverCardProps, 'body' | 'openDelay' | 'closeDelay'>> & {
  session: SessionInfo
  children: React.ReactElement
}) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)
  const triggerRef = useRef<HTMLElement | null>(null)
  const cardRef = useRef<HTMLDivElement | null>(null)
  const openTimer = useRef<number | null>(null)
  const closeTimer = useRef<number | null>(null)

  const cancelOpen = () => {
    if (openTimer.current !== null) {
      window.clearTimeout(openTimer.current)
      openTimer.current = null
    }
  }
  const cancelClose = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current)
      closeTimer.current = null
    }
  }
  const positionCard = (target: HTMLElement) => {
    const r = target.getBoundingClientRect()
    const wouldClipRight =
      r.right + VIEWPORT_EDGE_PADDING_PX + CARD_WIDTH_PX >
      window.innerWidth - VIEWPORT_EDGE_PADDING_PX
    const left = wouldClipRight
      ? Math.max(VIEWPORT_EDGE_PADDING_PX, r.left - CARD_WIDTH_PX - VIEWPORT_EDGE_PADDING_PX)
      : r.right + VIEWPORT_EDGE_PADDING_PX
    const top = Math.max(VIEWPORT_EDGE_PADDING_PX, r.top)
    setPos({ left, top })
  }
  const scheduleOpen = (target: HTMLElement) => {
    cancelClose()
    if (openTimer.current !== null) window.clearTimeout(openTimer.current)
    openTimer.current = window.setTimeout(() => {
      positionCard(target)
      setOpen(true)
      openTimer.current = null
    }, openDelay)
  }
  const scheduleClose = () => {
    cancelOpen()
    closeTimer.current = window.setTimeout(() => {
      setOpen(false)
      setPos(null)
      closeTimer.current = null
    }, closeDelay)
  }

  // Compose with any existing handlers the row already wired up —
  // the row passes its <button> with its own onClick / etc. We
  // need to fire our hover handlers in addition to whatever's
  // already there (the row uses onClick for resume, not onMouseEnter,
  // so there's no real conflict today, but the wiring is future-safe).
  const childProps = children.props as {
    'aria-describedby'?: string
    onBlur?: (e: React.FocusEvent<HTMLElement>) => void
    onClick?: (e: React.MouseEvent<HTMLElement>) => void
    onFocus?: (e: React.FocusEvent<HTMLElement>) => void
    onMouseDown?: (e: React.MouseEvent<HTMLElement>) => void
    onMouseEnter?: (e: React.MouseEvent<HTMLElement>) => void
    onMouseLeave?: (e: React.MouseEvent<HTMLElement>) => void
  }
  const prevEnter = childProps.onMouseEnter
  const prevLeave = childProps.onMouseLeave
  const prevFocus = childProps.onFocus
  const prevBlur = childProps.onBlur
  const prevDescribedBy = childProps['aria-describedby']

  const trigger = React.cloneElement(
    children as React.ReactElement<Record<string, unknown>>,
    {
      ref: (node: HTMLElement | null) => {
        triggerRef.current = node
      },
      onMouseEnter: (e: React.MouseEvent<HTMLElement>) => {
        scheduleOpen(e.currentTarget)
        prevEnter?.(e)
      },
      onMouseLeave: (e: React.MouseEvent<HTMLElement>) => {
        scheduleClose()
        prevLeave?.(e)
      },
      onFocus: (e: React.FocusEvent<HTMLElement>) => {
        scheduleOpen(e.currentTarget)
        prevFocus?.(e)
      },
      onBlur: (e: React.FocusEvent<HTMLElement>) => {
        // Don't close if focus moved into the card itself.
        const next = e.relatedTarget as Node | null
        if (next && cardRef.current?.contains(next)) return
        scheduleClose()
        prevBlur?.(e)
      },
      'aria-describedby': open ? 'session-hover-card' : prevDescribedBy,
    } as Record<string, unknown>
  )

  return (
    <>
      {trigger}
      {open && pos
        ? createPortal(
            <div
              aria-live="polite"
              className={cn(
                'fixed z-[220] w-[20rem] max-w-[calc(100vw-1rem)] rounded-md border border-(--ui-stroke-secondary) bg-(--ui-popover-background) px-3 py-2.5 text-xs text-(--ui-text-primary) shadow-lg',
                // Subtle fade-in (no slide — feels sluggish on a 200 ms trigger)
                'animate-in fade-in-0 duration-100'
              )}
              id="session-hover-card"
              onMouseEnter={() => cancelClose()}
              onMouseLeave={scheduleClose}
              ref={cardRef}
              role="tooltip"
              style={{ left: pos.left, top: pos.top }}
            >
              <p className="mb-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-(--ui-text-quaternary)">
                {session.title?.trim() || 'Session'}
              </p>
              <p className="text-[0.8125rem] leading-relaxed text-(--ui-text-secondary)">
                {body}
              </p>
              {session.summary_updated_at ? (
                <p className="mt-2 text-[0.6875rem] text-(--ui-text-quaternary)">
                  Updated {formatAgo(session.summary_updated_at)} · {session.message_count} messages
                  {session.summary_model ? ` · ${shortModel(session.summary_model)}` : ''}
                </p>
              ) : null}
            </div>,
            document.body
          )
        : null}
    </>
  )
}

/**
 * Tiny relative-time formatter — same shape as the existing age
 * formatter in SidebarSessionRow (which uses `formatAge` from
 * i18n). We keep this local because the hover card's "Updated 2
 * hours ago" footer is a single short string; pulling in the
 * full translations table here would balloon the bundle.
 */
function formatAgo(unixSeconds: number): string {
  const delta = Math.max(0, Date.now() / 1000 - unixSeconds)
  if (delta < 60) return 'just now'
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`
  if (delta < 86400) return `${Math.floor(delta / 3600)} h ago`
  return `${Math.floor(delta / 86400)} d ago`
}

function shortModel(model: string): string {
  // Trim long model strings for the footer (e.g. "openai/gpt-4o-mini-2024-07-18" → "gpt-4o-mini")
  const m = model.replace(/^[^/]+\//, '')
  return m.length > 24 ? `${m.slice(0, 22)}…` : m
}
