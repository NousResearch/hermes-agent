/** Shared kanban UI atoms: formatters, the identity avatar, the status menu,
 *  section chrome, and the masked scroller. Pure SDK + tokens. */

import {
  atom,
  coarseElapsed,
  Codicon,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  FadeScroll,
  profileColor,
  profileColorSoft,
  relativeTime,
  useQuery
} from '@hermes/plugin-sdk'
import { type ReactNode, useEffect, useState } from 'react'

import { fetchOrchestration, orchestrationKey, useKanbanScope } from './api'
import { columnLabel, useKanban } from './i18n'
import { columnMeta, type KanbanTask } from './types'

// Plugin-scoped i18n lives in ./i18n; re-exported so components import strings
// and chrome from one place (./ui).
export { columnHelp, columnLabel, type KanbanText, lockedReason, useKanban } from './i18n'

/** One-shot "open the new-task dialog in this lane" request, so a command that
 *  fires from ANYWHERE (keybind, palette) can reach the board page without the
 *  page having to exist yet: the handler navigates and drops the lane here, the
 *  page consumes it on arrival and clears it. Ephemeral by design — never
 *  persisted, so a remount can't reopen a dialog the user already dismissed. */
export const $newTaskLane = atom<null | string>(null)

/** Orchestration knobs (cached app-wide; the settings panel invalidates). */
export function useOrchestration() {
  const scope = useKanbanScope()

  return useQuery({ queryKey: orchestrationKey(scope), queryFn: fetchOrchestration, staleTime: 60_000 }).data
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

// ── card dependency rail (PR2) ───────────────────────────────────────────────
/** Rail tones: a parent that finished, one whose worker is live right now, and
 *  one that is neither (queued, blocked, or never started). */
export const DEP_TONES = {
  done: '#34d399',
  running: '#60a5fa',
  unfinished: 'var(--ui-text-quaternary)'
} as const

export type DepSegment = keyof typeof DEP_TONES

/** Which rail segment a parent's status earns. `archived` rides with `done`:
 *  an archived parent will never move again, so gating on it forever would be a
 *  deadlock rather than a dependency. */
export const depSegment = (status?: null | string): DepSegment =>
  status === 'done' || status === 'archived' ? 'done' : status === 'running' ? 'running' : 'unfinished'

export interface DependencyState {
  /** One entry per parent, in board order (empty when the card has none). */
  rail: { id: string; segment: DepSegment; title: string }[]
  /** True while at least one parent is unfinished — the card is gated. */
  waiting: boolean
  /** The unfinished parents: the rail's "and why". */
  blockedBy: { id: string; title: string }[]
  /** Child count, for the right-edge indicator. */
  children: number
}

/** Resolve a card's dependency state against the board's own task map. Parent
 *  statuses live on other cards, so the board owns the lookup, not the card. */
export function dependencyState(
  task: KanbanTask,
  lookup: (id: string) => undefined | { status?: null | string; title?: null | string }
): DependencyState {
  const rail = (task.links?.parents ?? []).map(id => {
    const parent = lookup(id)

    return { id, segment: depSegment(parent?.status), title: parent?.title?.trim() || shortId(id) }
  })

  const blockedBy = rail.filter(parent => parent.segment !== 'done').map(({ id, title }) => ({ id, title }))

  return {
    rail,
    waiting: blockedBy.length > 0,
    blockedBy,
    children: task.links?.children?.length ?? task.link_counts?.children ?? 0
  }
}

/** Right-edge indicator for a card that owns children: the children's aggregate
 *  state (live beats finished beats pending), or null when it has none. */
export function childrenIndicator(
  task: KanbanTask,
  lookup: (id: string) => undefined | { status?: null | string }
): null | DepSegment {
  const children = task.links?.children ?? []

  if (children.length === 0) {return null}
  const segments = children.map(id => depSegment(lookup(id)?.status))

  if (segments.includes('running')) {return 'running'}

  return segments.every(segment => segment === 'done') ? 'done' : 'unfinished'
}

// ── tenant (档案) colour coding (PR2) ────────────────────────────────────────
/** Twelve hues, interleaved so two adjacent entries never share a hue family.
 *  Every one clears 3:1 against both the light (#ffffff) and dark (#111113)
 *  surface — WCAG 1.4.11 non-text contrast, because the tint only ever fills
 *  the 3px bar, the tab dot and the rail segments; all text keeps its token. */
export const TENANT_HUES = [
  '#e11d48',
  '#0284c7',
  '#4d7c0f',
  '#9333ea',
  '#d97706',
  '#0f766e',
  '#1565c0',
  '#c2185b',
  '#0e7490',
  '#7c3aed',
  '#b45309',
  '#00897b'
] as const

/** FNV-1a with a murmur3 finalizer. The avalanche is load-bearing: plain FNV
 *  collides two of the five court profiles (gongbu/menxia) onto one hue. */
export const tenantHash = (tenant: string): number => {
  let hash = 0x811c9dc5

  for (let i = 0; i < tenant.length; i += 1) {
    hash = (hash ^ tenant.charCodeAt(i)) >>> 0
    hash = Math.imul(hash, 0x01000193) >>> 0
  }

  hash = (hash ^ (hash >>> 16)) >>> 0
  hash = Math.imul(hash, 0x7feb352d) >>> 0
  hash = (hash ^ (hash >>> 15)) >>> 0
  hash = Math.imul(hash, 0x846ca68b) >>> 0

  return (hash ^ (hash >>> 16)) >>> 0
}

/** The tenant's hue, or the neutral token for the un-tenanted lane ('' ⇒ '—'). */
export const tenantColor = (tenant?: null | string): string =>
  tenant ? TENANT_HUES[tenantHash(tenant) % TENANT_HUES.length] : 'var(--ui-text-quaternary)'

/** Tab-bar copy for a tenant: the empty tenant reads as an em dash. */
export const tenantLabel = (tenant: string): string => tenant || '—'

/** The tab row: "all" plus each distinct name, in board order. The "1 +
 *  distinct" count is the invariant worth holding here, so it lives somewhere a
 *  test can reach it. */
export const tenantTabList = (names: readonly string[]): string[] => ['', ...new Set(names)]

/**
 * Profile-tab scope for the board filter — '' means every card. A tab named
 * after a profile covers both places a card can carry that name: its tenant
 * (the board it belongs to) and its assignee (whose workload it is). Matching
 * tenant alone left the lane header saying "gongbu 2" while the gongbu tab
 * found nothing — the tab has to agree with the card's visible grouping.
 */
export const matchesProfileTab = (task: Pick<KanbanTask, 'assignee' | 'tenant'>, name: string): boolean =>
  !name || task.tenant === name || task.assignee === name

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

// ── card badges ──────────────────────────────────────────────────────────────

/** Compact "34m" for badge labels, off the same bucketing as run durations. */
export function fmtSecs(seconds: number): string {
  const { unit, value } = coarseElapsed(Math.max(0, Math.floor(seconds)) * 1000)

  return `${value}${ELAPSED_SUFFIX[unit]}`
}

/** A blocked card quiet for this long is stale — it needs a human. */
export const STALE_BLOCKED_SECONDS = 86_400

export interface RuntimeBadge {
  cap: number
  elapsed: number
  kind: 'near' | 'over'
}

/**
 * The card's runtime-cap badge state: `over` past the cap (the dispatcher will
 * time the run out), `near` past half of it, null with no cap or no run clock
 * (a queued card's cap is nobody's worry yet).
 */
export function runtimeCapBadge(task: KanbanTask, nowSecs: number): null | RuntimeBadge {
  const cap = task.max_runtime_seconds

  if (!cap || task.status !== 'running' || !task.started_at) {
    return null
  }

  const elapsed = Math.max(0, nowSecs - task.started_at)

  if (elapsed > cap) {
    return { cap, elapsed, kind: 'over' }
  }

  return elapsed > cap / 2 ? { cap, elapsed, kind: 'near' } : null
}

/** Blocked card that has seen no event for 24h+ — render the grey triage dot. */
export function staleBlocked(task: KanbanTask, nowSecs: number): boolean {
  return (
    task.status === 'blocked' &&
    typeof task.last_event_at === 'number' &&
    nowSecs - task.last_event_at > STALE_BLOCKED_SECONDS
  )
}

/** Ticking epoch-seconds clock for badges whose numbers age by the minute. */
export function useNowSecs(active: boolean): number {
  const [, force] = useState(0)

  useEffect(() => {
    if (!active) {
      return
    }

    const id = window.setInterval(() => force(n => n + 1), 5_000)

    return () => window.clearInterval(id)
  }, [active])

  return Math.floor(Date.now() / 1000)
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

// A short, edge-masked scroll area. Thin wrapper over the app's FadeScroll so
// the drawer's scrollers behave exactly like the ones in chat; kept as a local
// name because every call site here passes `max`.
export function ScrollFade({ children, deps, max = '9rem' }: { children: ReactNode; deps?: unknown; max?: string }) {
  return (
    <FadeScroll deps={deps} maxHeight={max}>
      {children}
    </FadeScroll>
  )
}
