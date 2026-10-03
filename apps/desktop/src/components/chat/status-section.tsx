import './status-stack.css'

import { type MouseEvent as ReactMouseEvent, type ReactNode, useRef, useState } from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'

interface StatusSectionProps {
  /** Optional right-aligned actions (text links / micro buttons). Pass
   *  `Button` with `size="micro"` + `variant="text"` or `"link"`. */
  accessory?: ReactNode
  children: ReactNode
  /** Optional inline status next to the label (running spinner, etc). */
  collapsedIndicator?: ReactNode
  defaultCollapsed?: boolean
  /** Compact live content stays visible while the full roster is collapsed. */
  preview?: ReactNode
  /** Optional glyph between the caret and the label (e.g. a `Codicon`). */
  icon?: ReactNode
  label: ReactNode
}

// A bottom-anchored group grows upward: expanding it lifts the header by the
// revealed rows' height and parks a row's dismiss control exactly where the
// header was. A second click at that same spot without moving the pointer is
// the stale collapse click, not intent for whatever slid under it. Anchoring on
// the expanding click's own position (not on a row's mount time) also keeps
// unrelated rows — a background task appearing while the group is open — from
// swallowing a click the user aimed at them. The window only has to cover the
// time a person needs to read the rows they just revealed; the position check
// does the real work, so it can be generous.
const STALE_REVEAL_CLICK_MS = 1500
const STALE_REVEAL_CLICK_PX = 4

/**
 * One collapsible group inside the composer status stack. Pure chrome — header
 * (caret + label) + body — styled to match the queue exactly so every status
 * (queue, subagents, background) reads as one piece. The stack supplies the
 * outer card and the dividers between groups; this owns only its own collapse.
 */
export function StatusSection({
  accessory,
  children,
  collapsedIndicator,
  defaultCollapsed = true,
  icon,
  label,
  preview
}: StatusSectionProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed)
  const revealedAt = useRef<{ at: number; x: number; y: number } | null>(null)

  const toggle = (event: ReactMouseEvent<HTMLButtonElement>) => {
    revealedAt.current = collapsed ? { at: performance.now(), x: event.clientX, y: event.clientY } : null
    setCollapsed(open => !open)
  }

  // Capture phase: stop the stale click before the control under it sees it and
  // honour the collapse the user actually asked for, instead of firing a
  // destructive action or silently dropping the click.
  const swallowStaleRevealClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    const anchor = revealedAt.current

    if (!anchor) {
      return
    }

    if (performance.now() - anchor.at > STALE_REVEAL_CLICK_MS) {
      return
    }

    if (Math.abs(event.clientX - anchor.x) > STALE_REVEAL_CLICK_PX) {
      return
    }

    if (Math.abs(event.clientY - anchor.y) > STALE_REVEAL_CLICK_PX) {
      return
    }

    if (!(event.target as Element | null)?.closest('[data-slot="status-dismiss"]')) {
      return
    }

    revealedAt.current = null
    event.preventDefault()
    event.stopPropagation()
    setCollapsed(true)
  }

  return (
    <div data-slot="status-section" onClickCapture={swallowStaleRevealClick}>
      <div className="status-section-header flex items-center gap-1 pr-1">
        <button
          aria-expanded={!collapsed}
          className="status-section-trigger flex min-w-0 flex-1 items-center gap-1.5 px-2 py-1 text-left text-xs font-normal text-muted-foreground/92 transition-colors hover:text-foreground/90"
          onClick={toggle}
          type="button"
        >
          <DisclosureCaret className="shrink-0" open={!collapsed} size="1em" />
          {icon && <span className="status-section-icon flex shrink-0 items-center">{icon}</span>}
          <span className="min-w-0 truncate">{label}</span>
          {collapsedIndicator && <span className="flex shrink-0 items-center">{collapsedIndicator}</span>}
        </button>
        {accessory && <div className="flex shrink-0 items-center gap-1">{accessory}</div>}
      </div>
      {(!collapsed || preview) && <div className="status-section-body">{collapsed ? preview : children}</div>}
    </div>
  )
}
