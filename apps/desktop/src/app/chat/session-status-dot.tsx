import { useStore } from '@nanostores/react'

import { type Translations, useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $backgroundRunningSessionIds } from '@/store/composer-status'
import { $unreadFinishedSessionIds } from '@/store/session'
import { $sessionColorById, sessionColorFor } from '@/store/session-color'
import { $attentionSessionIds, $stalledSessionIds, $workingSessionIds } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { type SessionDotState, sessionDotState } from './sidebar/session-row-state'

// A pure lookup table: each state maps to its className, aria-label, and title.
// No priority resolution here — sessionDotState already picked one. Label/title
// resolve from sidebar.row translations, keyed by name.
type DotVariant = {
  ariaLabel?: (r: Translations['sidebar']['row']) => string
  className: string
  role?: 'status'
  title?: (r: Translations['sidebar']['row']) => string
}

// Shared base for every active dot; idle is smaller and uses its own class.
const DOT_BASE = 'relative size-1.5 rounded-full'

// Pseudo-element ping ring that scales outward and fades — shared scaffold for
// the two pulsing dots. The `before:bg-*` color is written inline per variant
// (NOT interpolated here): Tailwind only generates utilities it can see as
// complete static strings, so a `before:bg-${color}` template never emits.
const PING = "before:absolute before:inset-0 before:animate-ping before:rounded-full before:content-['']"

const DOT_VARIANTS: Record<SessionDotState, DotVariant> = {
  // Amber steady — a clarify/approval is blocking the turn. Steady (not
  // pulsing) reads as "your turn", distinct from the accent pulse of a turn.
  'needs-input': {
    ariaLabel: r => r.needsInput,
    className: `${DOT_BASE} quest-glow bg-amber-500`,
    role: 'status',
    title: r => r.waitingForAnswer
  },
  // Accent pulse — the LLM turn is actively running.
  working: {
    ariaLabel: r => r.sessionRunning,
    className: `${DOT_BASE} bg-(--ui-accent) shadow-[0_0_0.625rem_color-mix(in_srgb,var(--ui-accent)_55%,transparent)] ${PING} before:bg-(--ui-accent) before:opacity-70`,
    role: 'status'
  },
  // Quiet accent pulse — the turn is still authoritative-running, but no
  // stream activity has arrived for the watchdog window.
  stalled: {
    ariaLabel: r => r.sessionRunning,
    className: `${DOT_BASE} bg-(--ui-accent) opacity-70 ${PING} before:bg-(--ui-accent) before:opacity-40`,
    role: 'status',
    title: r => r.sessionRunning
  },
  // Pulsing gray — a terminal(background=true) process is alive while the LLM
  // is idle. Gray (not accent) reads as "something chugging along". Brighter
  // than muted-foreground so it's visible against the surface.
  background: {
    ariaLabel: r => r.backgroundRunning,
    className: `${DOT_BASE} bg-muted-foreground/80 ${PING} before:bg-muted-foreground/80 before:opacity-60`,
    role: 'status',
    title: r => r.backgroundRunning
  },
  // Steady green — a background session's turn completed and the user hasn't
  // opened it since. "Something new here, go look."
  unread: {
    ariaLabel: r => r.finishedUnread,
    className: `${DOT_BASE} bg-emerald-500`,
    role: 'status',
    title: r => r.finishedUnread
  },
  idle: {
    className: 'size-1 rounded-full bg-(--ui-text-quaternary) opacity-80'
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
 * SESSION STATUS DOT — the ONE primitive both the sidebar row and the pane tab
 * render, so a session's status/color can never disagree between the two
 * surfaces. It reads every signal itself from the shared stores keyed by the
 * stored session id: live state (working / needs-input / stalled / unread /
 * background, mutually exclusive via `sessionDotState`) and the resolved color
 * (override → project color, via `sessionColorFor`). An idle session shows its
 * project color; the active states own the dot with their semantic color so an
 * attention cue is never masked by the inherited tint.
 */
export function SessionStatusDot({ storedSessionId, session, branchStem, className }: SessionStatusDotProps) {
  const { t } = useI18n()
  const r = t.sidebar.row

  // Subscribe to the shared color map for reactivity; sessionColorFor falls
  // back to the resolver for a session outside the recents page.
  useStore($sessionColorById)
  const color = sessionColorFor(session) ?? null

  const needsInput = useStore($attentionSessionIds).includes(storedSessionId)
  const isWorking = useStore($workingSessionIds).includes(storedSessionId)
  const isStalled = useStore($stalledSessionIds).includes(storedSessionId)
  const isUnread = useStore($unreadFinishedSessionIds).includes(storedSessionId)
  const hasBackground = useStore($backgroundRunningSessionIds).includes(storedSessionId)

  const dotState = sessionDotState({ hasBackground, isStalled, isUnread, isWorking, needsInput })

  return (
    <span className={cn('flex items-center gap-0.5', className)}>
      {branchStem ? (
        <span aria-hidden className="shrink-0 font-mono text-[0.625rem] leading-none text-(--ui-text-quaternary)">
          {branchStem}
        </span>
      ) : null}
      {dotState === 'idle' && color ? (
        <span aria-hidden="true" className="size-1 rounded-full" style={{ backgroundColor: color }} />
      ) : (
        <span
          aria-label={DOT_VARIANTS[dotState].ariaLabel?.(r)}
          className={DOT_VARIANTS[dotState].className}
          role={DOT_VARIANTS[dotState].role}
          title={DOT_VARIANTS[dotState].title?.(r)}
        />
      )}
    </span>
  )
}
