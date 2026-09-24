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

// Shared size for active marks; state-specific shape is part of the signal.
const DOT_SIZE = 'size-1.5'

// Color is supplementary: shape + fill keep the important states distinct for
// color-blind users. Motion on a tiny mark only says "something is happening"
// — the row's arc already carries that signal better — so these marks stay still.
const DOT_VARIANTS: Record<SessionDotState, DotVariant> = {
  // Amber — a clarify/approval is blocking the turn. The one "act now" color,
  // and the only state the user is required to do something about.
  'needs-input': {
    ariaLabel: r => r.needsInput,
    className: `${DOT_SIZE} rotate-45 rounded-[1px] bg-(--ui-yellow)`,
    role: 'status',
    title: r => r.waitingForAnswer
  },
  // Accent — the turn is running. The row's arc carries the motion.
  working: {
    ariaLabel: r => r.sessionRunning,
    className: `${DOT_SIZE} rounded-full bg-(--ui-accent)`,
    role: 'status'
  },
  // Hollow accent — still authoritatively running, but nothing has arrived for
  // the watchdog window. Same color as working because it IS working; hollow
  // because nothing is coming out of it right now.
  stalled: {
    ariaLabel: r => r.sessionRunning,
    className: `${DOT_SIZE} rounded-full border border-(--ui-accent)`,
    role: 'status',
    title: r => r.sessionRunning
  },
  // Hollow muted — a terminal(background=true) process outlived the turn. An
  // outline reads as "still open" without claiming the model is working; a
  // filled grey dot read as finished, the opposite of what this means.
  background: {
    ariaLabel: r => r.backgroundRunning,
    className: `${DOT_SIZE} rounded-[1px] border border-(--ui-text-tertiary)`,
    role: 'status',
    title: r => r.backgroundRunning
  },
  // Emerald — the turn finished while the user was looking elsewhere. The
  // color is theme-derived (`--ui-success`, a success green rotated toward the
  // accent) so eight finished dots can't sit in the sidebar fighting a palette
  // they don't belong to. Under a green accent it stays emerald.
  unread: {
    ariaLabel: r => r.finishedUnread,
    className: `${DOT_SIZE} rounded-[1px] bg-(--ui-success)`,
    role: 'status',
    title: r => r.finishedUnread
  },
  // Hollow grey, the faintest ink the app has — nothing has ever run here. It
  // shares the outline with `background` because both mean "open, not
  // producing", and sits a shade dimmer because a draft is the one state that
  // has yet to do anything at all.
  draft: {
    ariaLabel: r => r.draftSession,
    className: `${DOT_SIZE} rounded-full border border-(--ui-text-quaternary)`,
    title: r => r.draftSession
  },
  // Settled: the project color when there is one, else the faintest filled
  // grey. Every session shows SOME mark — a row with nothing in the lead slot
  // reads as broken next to its neighbours, so "no color" falls back to the
  // quietest ink rather than to an invisible dot.
  idle: {
    className: 'size-1 rounded-full bg-(--ui-text-quaternary)'
  }
}

/** The dot a state paints, for surfaces that describe a status rather than
 *  render a session — the sidebar's status filter, say. Idle carries no color
 *  of its own (it inherits the project's), so callers supply one. */
export const sessionDotClassName = (state: SessionDotState): string => DOT_VARIANTS[state].className

export interface SessionStatusDotProps {
  /** The STORED session id — the key every live-state atom (working /
   *  attention / stalled / unread / background) is keyed by, on BOTH surfaces:
   *  the sidebar row's `session.id` and a pane tile's `storedSessionId` are the
   *  same stored id (`$workingSessionIds` et al. map `storedSessionId`).
   *
   *  Null on a new chat that has yet to reach the backend — no id to key by,
   *  and no turn behind it, which is the draft state by definition. */
  storedSessionId: null | string
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
  const dotState = useStoreSelector($sessionDotStateById, states =>
    storedSessionId ? (states[storedSessionId] ?? 'idle') : 'draft'
  )

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
