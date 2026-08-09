import { useStore } from '@nanostores/react'

import { type Translations, useI18n } from '@/i18n'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $sessionColorById, sessionColorFor } from '@/store/session-color'
import { $sessionDotStateById, type SessionDotState } from '@/store/session-dot-state'
import type { SessionInfo } from '@/types/hermes'

// A pure lookup table: each state maps to its className, aria-label, and title.
// No priority resolution here — $sessionDotStateById already picked one.
// Label/title resolve from sidebar.row translations, keyed by name.
type DotVariant = {
  ariaLabel?: (r: Translations['sidebar']['row']) => string
  className: string
  role?: 'status'
  title?: (r: Translations['sidebar']['row']) => string
}

// Shared base for every active dot; idle is smaller and uses its own class.
const DOT_BASE = 'size-1.5 rounded-full'

// Three colors and one fill/hollow axis, none of it moving. Motion on a 6px
// circle can only say "something is happening" — which the row's arc already
// says, better — while costing a repaint per frame on every row at once. What
// the dot is for is telling states APART, and that is a job for color and fill:
// filled means producing, hollow means open but quiet. The two states this
// replaces differed by 30% opacity and were, in practice, the same dot.
const DOT_VARIANTS: Record<SessionDotState, DotVariant> = {
  // Amber — a clarify/approval is blocking the turn. The one "act now" color,
  // and the only state the user is required to do something about.
  'needs-input': {
    ariaLabel: r => r.needsInput,
    className: `${DOT_BASE} bg-amber-500`,
    role: 'status',
    title: r => r.waitingForAnswer
  },
  // Accent — the turn is running. The row's arc carries the motion.
  working: {
    ariaLabel: r => r.sessionRunning,
    className: `${DOT_BASE} bg-(--ui-accent)`,
    role: 'status'
  },
  // Hollow accent — still authoritatively running, but nothing has arrived for
  // the watchdog window. Same color as working because it IS working; hollow
  // because nothing is coming out of it right now.
  stalled: {
    ariaLabel: r => r.sessionRunning,
    className: `${DOT_BASE} border border-(--ui-accent)`,
    role: 'status',
    title: r => r.sessionRunning
  },
  // Hollow muted — a terminal(background=true) process outlived the turn. An
  // outline reads as "still open" without claiming the model is working; a
  // filled grey dot read as finished, the opposite of what this means.
  background: {
    ariaLabel: r => r.backgroundRunning,
    className: `${DOT_BASE} border border-(--ui-text-tertiary)`,
    role: 'status',
    title: r => r.backgroundRunning
  },
  // Emerald — the turn finished while the user was looking elsewhere.
  unread: {
    ariaLabel: r => r.finishedUnread,
    className: `${DOT_BASE} bg-emerald-500`,
    role: 'status',
    title: r => r.finishedUnread
  },
  // Settled: the project color, or nothing at all. An uncolored session used to
  // get a grey dot, which put a mark of the same weight as a status next to
  // every resting row and made "no color" look like a state of its own.
  idle: {
    className: 'size-1 rounded-full'
  }
}

export interface SessionStatusDotProps {
  /** The STORED session id — the key every live-state atom (working /
   *  attention / stalled / unread / background) is keyed by, on BOTH surfaces:
   *  the sidebar row's `session.id` and a pane tile's `storedSessionId` are the
   *  same stored id (`$workingSessionIds` et al. map `storedSessionId`). */
  storedSessionId: string
  /** The session row for color resolution — recents OR the project tree. Both
   *  call sites already hold it; passing it lets the idle dot inherit the
   *  project color even for a session older than the paginated recents page
   *  (which has no `$sessionColorById` entry). */
  session?: null | SessionInfo
  /** TUI-style tree stem for a branched session (`└─ ` / `├─ `). */
  branchStem?: string
  /** Applied to the OUTER wrapper (stem + dot) — e.g. hover-fade on the
   *  reorder handle. */
  className?: string
}

/**
 * SESSION STATUS DOT — the ONE primitive the sidebar row, the pane tabs, and
 * the session switcher render, so a session's status can never disagree
 * between surfaces. It resolves everything itself from the stored session id:
 * the live state (via `$sessionDotStateById`, already reduced to one mutually
 * exclusive answer) and the color (override → project, via `sessionColorFor`).
 * An idle session shows its project color; the active states own the dot with
 * their semantic color so an attention cue is never masked by the tint.
 */
export function SessionStatusDot({ storedSessionId, session, branchStem, className }: SessionStatusDotProps) {
  const { t } = useI18n()
  const r = t.sidebar.row

  // Subscribe to the shared color map for reactivity; sessionColorFor falls
  // back to the resolver for a session outside the recents page.
  useStore($sessionColorById)
  const color = sessionColorFor(session) ?? null

  // Selector, not a plain useStore: the map is rebuilt whenever any session's
  // status changes, but a given dot only repaints when ITS OWN state flips.
  const dotState = useStoreSelector($sessionDotStateById, states => states[storedSessionId] ?? 'idle')
  const variant = DOT_VARIANTS[dotState]

  return (
    <span className={cn('flex items-center gap-0.5', className)}>
      {branchStem ? (
        <span aria-hidden className="shrink-0 font-mono text-[0.625rem] leading-none text-(--ui-text-quaternary)">
          {branchStem}
        </span>
      ) : null}
      {dotState === 'idle' ? (
        // Rendered even with no color to paint: an empty dot of the same size
        // keeps every row's title on one left edge, so a session finishing
        // can't shift the list under the pointer.
        <span aria-hidden="true" className={variant.className} style={color ? { backgroundColor: color } : undefined} />
      ) : (
        <span
          aria-label={variant.ariaLabel?.(r)}
          className={variant.className}
          role={variant.role}
          title={variant.title?.(r)}
        />
      )}
    </span>
  )
}
