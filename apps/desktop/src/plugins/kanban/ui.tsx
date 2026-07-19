/** Shared kanban UI atoms: formatters, the identity avatar, the status menu,
 *  section chrome, and the masked scroller. Pure SDK + tokens. */

import {
  coarseElapsed,
  Codicon,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  profileColor,
  profileColorSoft,
  relativeTime,
  useQuery
} from '@hermes/plugin-sdk'
import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { fetchOrchestration, ORCHESTRATION_KEY } from './api'
import { columnLabel, useKanban } from './i18n'
import { columnMeta, type KanbanTask } from './types'

// Plugin-scoped i18n lives in ./i18n; re-exported so components import strings
// and chrome from one place (./ui).
export { columnHelp, columnLabel, type KanbanText, lockedReason, useKanban } from './i18n'

/** Orchestration knobs (cached app-wide; the settings panel invalidates). */
export function useOrchestration() {
  return useQuery({ queryKey: ORCHESTRATION_KEY, queryFn: fetchOrchestration, staleTime: 60_000 }).data
}

/** The dispatcher's configured fallback for unassigned ready cards
 *  (`kanban.default_assignee`) — '' when unset, i.e. unassigned never runs. */
export function useDefaultAssignee(): string {
  return useOrchestration()?.default_assignee.trim() ?? ''
}

// System-owned drop targets — you can drag a card OUT of these, never INTO
// them, so lanes/menus must not offer them as targets. `running`/`review` are
// claimed by the dispatcher; `scheduled` needs a wake-up time only an agent or
// the CLI can attach (a bare status drag is refused with a 409). The reason
// copy lives in the plugin i18n bundle (`locked.*`); see `lockedReason`.
export const LOCKED_COLUMNS = ['review', 'running', 'scheduled'] as const

export const isLockedTarget = (name: string): boolean => (LOCKED_COLUMNS as readonly string[]).includes(name)

export const shortId = (id?: null | string) => (id ?? '').replace(/^t_/, '').slice(0, 6)

// The electron REST bridge throws `Error("409: {\"detail\":\"…\"}")`; pull out
// the human-readable detail for a toast.
export function errText(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err)
  const brace = raw.indexOf('{')

  if (brace !== -1) {
    try {
      return (JSON.parse(raw.slice(brace)) as { detail?: string }).detail ?? raw
    } catch {
      // Not JSON — fall through to the raw message.
    }
  }

  return raw
}

/** Backend timestamps are epoch SECONDS; the canonical formatter takes ms. */
export const ago = (seconds?: null | number): null | string => (seconds ? relativeTime(seconds * 1000) : null)

const ELAPSED_SUFFIX = { day: 'd', hour: 'h', minute: 'm', second: 's' } as const

/** Compact run duration ("42s", "3m") off the canonical elapsed bucketing. */
export function duration(start?: null | number, end?: null | number): null | string {
  if (!start || !end || end < start) {
    return null
  }

  const { unit, value } = coarseElapsed((end - start) * 1000)

  return `${value}${ELAPSED_SUFFIX[unit]}`
}

// ── liveness ─────────────────────────────────────────────────────────────────

/** Live elapsed label ("34s", "2m") that keeps ticking while mounted. */
function useTicking(start?: null | number): null | string {
  const [, force] = useState(0)

  useEffect(() => {
    if (!start) {
      return
    }

    const id = window.setInterval(() => force(n => n + 1), 5_000)

    return () => window.clearInterval(id)
  }, [start])

  if (!start) {
    return null
  }

  const { unit, value } = coarseElapsed(Math.max(0, Date.now() - start * 1000))

  return `${value}${ELAPSED_SUFFIX[unit]}`
}

export type ArcState = 'queued' | 'running' | 'stale'

/**
 * The card's machine-activity state. The board looked dead between "I made a
 * card" and "it's suddenly running" — this narrates the in-between. Only the
 * working states animate the border arc (see kanban.css): running = brisk
 * sweep, no-heartbeat = amber crawl. `queued` (triage / assigned-ready /
 * review) renders as the footer's named-agent chip — motion means work.
 */
export function arcState(task: KanbanTask, fallbackAssignee: string): ArcState | null {
  if (task.status === 'running') {
    // No heartbeat for 2+ min = the worker may have died; the dispatcher will
    // reclaim it, but be honest instead of sweeping green forever.
    const stale = task.last_heartbeat_at ? Date.now() / 1000 - task.last_heartbeat_at > 120 : false

    return stale ? 'stale' : 'running'
  }

  const queued =
    task.status === 'triage' ||
    task.status === 'review' ||
    (task.status === 'ready' && Boolean(task.assignee || fallbackAssignee))

  return queued ? 'queued' : null
}

/** Ticking "working · 34s" line for running cards (elapsed since claim). */
export function RunClock({ task }: { task: KanbanTask }) {
  const k = useKanban()
  const elapsed = useTicking(task.started_at)

  if (!elapsed) {
    return null
  }

  return (
    <span className="shrink-0 font-medium" style={{ color: columnMeta('running').tone }}>
      {k.working} · {elapsed}
    </span>
  )
}

function initials(name: string): string {
  const parts = name
    .trim()
    .split(/[\s_\-./]+/)
    .filter(Boolean)

  return `${parts[0]?.[0] ?? '?'}${parts[1]?.[0] ?? ''}`.toUpperCase()
}

export function Avatar({ name, size = '1.25rem' }: { name: string; size?: string }) {
  // Same identity hue the rest of the app uses (profileColor); default/empty
  // profiles are neutral. Soft tag fill + colored glyph, per the app's tags.
  const color = profileColor(name)

  return (
    <span
      className="grid shrink-0 place-items-center rounded-full font-semibold"
      style={{
        backgroundColor: color ? profileColorSoft(color, 22) : 'var(--ui-bg-quaternary)',
        color: color ?? 'var(--ui-text-secondary)',
        fontSize: '0.5625rem',
        height: size,
        width: size
      }}
      title={name}
    >
      {initials(name)}
    </span>
  )
}

// Jira-style status control: a colored button showing the current state, click
// to transition. Options carry their column dot; the active one is checked.
export function StatusMenu({
  columns,
  onMove,
  status
}: {
  columns: string[]
  onMove: (status: string) => void
  status: string
}) {
  const k = useKanban()
  const meta = columnMeta(status)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className="inline-flex items-center gap-1.5 rounded px-2 py-1 text-[0.6875rem] font-semibold uppercase tracking-wide transition-[filter] hover:brightness-105"
          style={{ backgroundColor: `color-mix(in srgb, ${meta.tone} 15%, transparent)`, color: meta.tone }}
          type="button"
        >
          <span className="size-1.5 rounded-full" style={{ backgroundColor: meta.tone }} />
          {columnLabel(k, status)}
          <Codicon name="chevron-down" size="0.7rem" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {columns
          .filter(name => name === status || !isLockedTarget(name))
          .map(name => (
            <DropdownMenuItem key={name} onSelect={() => onMove(name)}>
              <span className="size-2 rounded-full" style={{ backgroundColor: columnMeta(name).tone }} />
              {columnLabel(k, name)}
              {name === status && <Codicon className="ml-auto" name="check" size="0.8rem" />}
            </DropdownMenuItem>
          ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// The board's one field/section-label style — hoisted so Section (here), the
// create dialog's Field, and the orchestration panel all read identically.
export const FIELD_LABEL = 'text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-(--ui-text-quaternary)'

export function Section({ action, children, label }: { action?: ReactNode; children: ReactNode; label: string }) {
  return (
    <section className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <div className={FIELD_LABEL}>{label}</div>
        {action}
      </div>
      {children}
    </section>
  )
}

// Tinted advisory panel: a `tone`-washed body with a matching left rule and a
// tone-colored icon+title header. Shared by the drawer's diagnostics and its
// ready-but-unassigned warning so both read identically.
export function Callout({
  children,
  icon = 'warning',
  title,
  tone
}: {
  children?: ReactNode
  icon?: string
  title: ReactNode
  tone: string
}) {
  return (
    <div
      className="flex flex-col gap-2 rounded-md p-2.5"
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

// A short, edge-masked scroll area. The fades are EDGE-AWARE like the rest of
// the app: a gradient only appears on a side that actually has clipped content
// (nothing to scroll → no mask at all), tracked via scroll + resize. Plus
// `overscroll-contain` so scrolling it never chains into the drawer. When
// `deps` is provided it re-pins to the bottom on change — the activity feed's
// newest-at-bottom behavior.
export function ScrollFade({ children, deps, max = '9rem' }: { children: ReactNode; deps?: unknown; max?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [edges, setEdges] = useState({ above: false, below: false })

  const measure = () => {
    const el = ref.current

    if (!el) {
      return
    }

    const above = el.scrollTop > 1
    const below = el.scrollTop + el.clientHeight < el.scrollHeight - 1

    setEdges(prev => (prev.above === above && prev.below === below ? prev : { above, below }))
  }

  useLayoutEffect(() => {
    if (deps !== undefined && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight
    }

    measure()
  }, [deps])

  useLayoutEffect(() => {
    const el = ref.current

    if (!el) {
      return
    }

    const observer = new ResizeObserver(measure)
    observer.observe(el)

    return () => observer.disconnect()
  }, [])

  const stops = [
    edges.above ? 'transparent, black 1.25rem' : 'black',
    edges.below ? 'calc(100% - 1.25rem), transparent' : 'black'
  ]

  const mask = `linear-gradient(to bottom, ${stops[0]}, black ${stops[1]})`

  return (
    <div
      className="overflow-y-auto overscroll-contain"
      onScroll={measure}
      ref={ref}
      style={
        edges.above || edges.below ? { maskImage: mask, maxHeight: max, WebkitMaskImage: mask } : { maxHeight: max }
      }
    >
      {children}
    </div>
  )
}
