/**
 * The "Output log" drawer section: a collapsed disclosure row that expands into
 * the card worker's live transcript.
 *
 * Collapsed it shows one line (worker state + message count). Expanded it shows
 * the feed in time order — assistant text, tool calls with a one-line argument
 * summary, tool results, and thinking blocks when the model emitted them —
 * newest at the bottom, auto-following unless the reader has scrolled up. While
 * the card is `running` it polls every few seconds; a finished card shows its
 * whole transcript and stops polling. A card with no worker session says so.
 */

import {
  Badge,
  Button,
  cn,
  Codicon,
  host,
  Loader,
  Tip,
  useQuery,
  type KanbanCardDrawerSectionProps
} from '@hermes/plugin-sdk'
import { type ReactNode, type RefObject, useCallback, useEffect, useMemo, useRef, useState } from 'react'

// ---------------------------------------------------------------------------
// Backend binding — `ctx.rest` is handed in at register() time (see plugin.tsx).
// ---------------------------------------------------------------------------

type RestCall = <T>(path: string, opts?: { body?: unknown; method?: string }) => Promise<T>

let restCall: RestCall | null = null

/** Bind the plugin's namespace-scoped REST door (called from `register`). */
export function bindOutputLogRest(rest: RestCall): void {
  restCall = rest
}

// ---------------------------------------------------------------------------
// Wire types (mirror plugins/kanban-output-log/dashboard/plugin_api.py)
// ---------------------------------------------------------------------------

type OutputState = 'no_db' | 'no_messages' | 'no_session' | 'ok' | 'session_missing'

interface OutputEntry {
  id: number
  kind: string
  role: string
  seq: number
  text: string
  tool_name?: null | string
  ts: number
}

interface OutputResponse {
  entries: OutputEntry[]
  has_earlier: boolean
  last_at: null | number
  next_before: null | number
  profile: null | string
  returned: number
  session_id: null | string
  started_at: null | number
  state: OutputState
  task_id: string
  task_status?: string
  total: number
}

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

const POLL_RUNNING_MS = 5_000
const TAIL_ROWS = 200
/** Bound paging into earlier rows; the live tail is separately bounded by TAIL_ROWS.
 *  Older rows stop loading at this budget (growth on an active run may exceed it). */
const MAX_ROWS_HELD = 800

const OPEN_STORAGE_KEY = 'hermes.kanbanOutputLog.open'

// Module-level so the disclosure state survives the drawer closing and
// reopening within a session (localStorage carries it across app restarts).
let openMemo: boolean | null = null

function readOpen(): boolean {
  if (openMemo !== null) {
    return openMemo
  }

  try {
    openMemo = localStorage.getItem(OPEN_STORAGE_KEY) === '1'
  } catch {
    openMemo = false
  }

  return openMemo
}

function persistOpen(value: boolean): void {
  openMemo = value

  try {
    localStorage.setItem(OPEN_STORAGE_KEY, value ? '1' : '0')
  } catch {
    // private mode / storage disabled — the module-level memo still holds this session
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function dedupe(entries: readonly OutputEntry[]): OutputEntry[] {
  const seen = new Set<string>()
  const out: OutputEntry[] = []

  for (const entry of entries) {
    const key = `${entry.id}.${entry.seq}`

    if (!seen.has(key)) {
      seen.add(key)
      out.push(entry)
    }
  }

  out.sort((a, b) => a.id - b.id || a.seq - b.seq)

  return out
}

function clockTime(ts: null | number): string {
  if (!ts) {
    return ''
  }

  try {
    return new Date(ts * 1000).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  } catch {
    return ''
  }
}

async function fetchOutput(board: null | string, taskId: string, before?: number): Promise<OutputResponse> {
  if (!restCall) {
    throw new Error('kanban-output-log: backend unavailable')
  }

  const params = new URLSearchParams({ tail: String(TAIL_ROWS) })

  if (board) {
    params.set('board', board)
  }

  if (before) {
    params.set('before', String(before))
  }

  return restCall<OutputResponse>(`/tasks/${encodeURIComponent(taskId)}/output?${params.toString()}`)
}

function stateSummary(data: OutputResponse | undefined, error: unknown, running: boolean, status: string): string {
  if (error) {
    return 'Output unavailable'
  }

  if (!data) {
    return 'Loading…'
  }

  switch (data.state) {
    case 'no_session':
      return 'No output yet — no worker has run for this card'
    case 'session_missing':
      return 'Worker session no longer in the profile’s state.db'
    case 'no_db':
      return 'Profile state.db unavailable'
    case 'no_messages':
      return 'Running — no messages recorded yet'
    case 'ok': {
      const label = running ? 'running' : status || (data.task_status ?? '')
      const lines = data.returned

      return `${label}${label ? ' · ' : ''}${lines} ${lines === 1 ? 'line' : 'lines'} · ${data.total} messages`
    }
    default:
      return status
  }
}

// ---------------------------------------------------------------------------
// Feed line
// ---------------------------------------------------------------------------

const KIND_ICON: Record<string, string> = {
  summary: 'fold',
  text: 'comment',
  thinking: 'lightbulb',
  tool_call: 'chevron-right',
  tool_result: 'arrow-left',
  user: 'account'
}

function FeedLine({ entry }: { entry: OutputEntry }): ReactNode {
  const icon = KIND_ICON[entry.kind] ?? 'circle-small-filled'

  return (
    <li
      className={cn(
        'flex items-start gap-1.5 border-l-2 py-0.5 pl-2 text-[0.6875rem] leading-snug',
        entry.kind === 'thinking' && 'border-transparent text-(--ui-text-quaternary) italic',
        entry.kind === 'text' && 'border-transparent text-(--ui-text-secondary)',
        entry.kind === 'tool_call' && 'border-(--ui-stroke-tertiary) text-(--ui-text-secondary)',
        entry.kind === 'tool_result' && 'border-(--ui-stroke-tertiary) text-(--ui-text-tertiary)',
        (entry.kind === 'user' || entry.kind === 'system' || entry.kind === 'summary') &&
          'border-transparent text-(--ui-text-tertiary)'
      )}
      data-kind={entry.kind}
    >
      <Codicon
        className="mt-0.5 shrink-0 opacity-70"
        name={icon}
        size="0.7rem"
      />
      <div className="min-w-0 flex-1">
        {entry.kind === 'tool_call' && (
          <span className="mr-1 font-mono font-medium text-(--ui-text-primary)">{entry.tool_name ?? 'tool'}</span>
        )}
        {entry.kind === 'tool_result' && entry.tool_name && (
          <span className="mr-1 font-mono opacity-70">{entry.tool_name}</span>
        )}
        <span className="[overflow-wrap:anywhere] whitespace-pre-wrap">{entry.text || '—'}</span>
      </div>
      <span className="shrink-0 text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">
        {clockTime(entry.ts)}
      </span>
    </li>
  )
}

// ---------------------------------------------------------------------------
// The expanded panel (feed + footer), and the section that discloses it
// ---------------------------------------------------------------------------

interface OutputLogFooterProps {
  atCap: boolean
  canLoadEarlier: boolean
  data: undefined | OutputResponse
  loadingEarlier: boolean
  moreEarlier: boolean
  onLoadEarlier: () => void
}

/** The line under the feed: whose session this is, and the pager. */
function OutputLogFooter({
  atCap,
  canLoadEarlier,
  data,
  loadingEarlier,
  moreEarlier,
  onLoadEarlier
}: OutputLogFooterProps): ReactNode {
  return (
    <div className="flex items-center gap-2 border-t border-(--ui-stroke-tertiary) px-2 py-1 text-[0.625rem] text-(--ui-text-quaternary)">
      <Badge size="xs" variant="muted">
        {data?.session_id ? data.session_id.slice(0, 18) : 'session'}
      </Badge>
      {data?.profile && <span>{data.profile}</span>}
      {data?.started_at && <span>started {clockTime(data.started_at)}</span>}
      {data?.last_at && <span>last {clockTime(data.last_at)}</span>}
      <span className="ml-auto flex items-center gap-2">
        {atCap && <span>holding last {MAX_ROWS_HELD}</span>}
        <Tip label={moreEarlier ? 'Load older transcript lines' : 'Reached the start of the transcript'}>
          <Button
            disabled={!canLoadEarlier || loadingEarlier}
            onClick={onLoadEarlier}
            size="sm"
            type="button"
            variant="outline"
          >
            <Codicon name="history" size="0.75rem" />
            {loadingEarlier ? 'Loading…' : 'Load earlier'}
          </Button>
        </Tip>
      </span>
    </div>
  )
}

interface OutputLogPanelProps extends OutputLogFooterProps {
  error: unknown
  onScroll: () => void
  rows: OutputEntry[]
  scrollRef: RefObject<HTMLDivElement | null>
}

/** What the section shows when it is open: the transcript feed, or why there is none. */
function OutputLogPanel({
  atCap,
  canLoadEarlier,
  data,
  error,
  loadingEarlier,
  moreEarlier,
  onLoadEarlier,
  onScroll,
  rows,
  scrollRef
}: OutputLogPanelProps): ReactNode {
  const body = data?.state === 'ok' ? null : data?.state === 'no_session'
    ? 'No output yet — this card has no worker session (queued, or never started).'
    : error
      ? 'Could not load the worker output.'
      : ''

  return (
    <div className="flex flex-col gap-2 rounded-md border border-(--ui-stroke-tertiary)">
      {body !== null ? (
        <p className="px-3 py-3 text-[0.6875rem] text-(--ui-text-tertiary)" data-slot="kanban-output-empty">
          {body}
        </p>
      ) : (
        <>
          <div
            className="max-h-[18rem] overflow-y-auto px-2 py-1.5"
            data-slot="kanban-output-scroll"
            onScroll={onScroll}
            ref={scrollRef}
          >
            <ul className="flex flex-col gap-0.5">
              {rows.map(entry => (
                <FeedLine entry={entry} key={`${entry.id}.${entry.seq}`} />
              ))}
            </ul>
          </div>
          <OutputLogFooter
            atCap={atCap}
            canLoadEarlier={canLoadEarlier}
            data={data}
            loadingEarlier={loadingEarlier}
            moreEarlier={moreEarlier}
            onLoadEarlier={onLoadEarlier}
          />
        </>
      )}
    </div>
  )
}

/** Whether the pager is worth offering: a real transcript, more rows behind it, and not
 *  already holding the cap. */
function canPageBack(
  state: OutputState | undefined,
  moreEarlier: boolean,
  atCap: boolean,
  minId: null | number
): boolean {
  return state === 'ok' && moreEarlier && !atCap && minId !== null
}

// ---------------------------------------------------------------------------
// The section
// ---------------------------------------------------------------------------

export function OutputLogSection({ board, running, status, taskId }: KanbanCardDrawerSectionProps): ReactNode {
  const [open, setOpen] = useState<boolean>(readOpen)
  const [earlier, setEarlier] = useState<OutputEntry[]>([])
  // `null` until a page has been loaded: before that the server's own `has_earlier` decides
  // whether a pager is offered, so a transcript that already fits shows the button disabled
  // instead of offering a page that can only come back empty.
  const [olderAvailable, setOlderAvailable] = useState<boolean | null>(null)
  const [loadingEarlier, setLoadingEarlier] = useState(false)
  const [pinned, setPinned] = useState(true)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const lastTaskRef = useRef<string | null>(null)
  const previousRunRef = useRef<{ board: null | string; running: boolean; taskId: string } | null>(null)

  const { data, error, isFetching, refetch } = useQuery<OutputResponse>({
    enabled: restCall !== null,
    queryFn: () => fetchOutput(board, taskId),
    queryKey: ['kanban-output-log', board ?? '', taskId],
    refetchInterval: running ? POLL_RUNNING_MS : false,
    refetchOnWindowFocus: false
  })

  // The last poll can precede completion; fetch once more before polling stops.
  useEffect(() => {
    const previous = previousRunRef.current
    previousRunRef.current = { board, running, taskId }
    if (previous?.running && !running && previous.board === board && previous.taskId === taskId) {
      void refetch()
    }
  }, [board, running, taskId, refetch])

  // A different card in the same drawer: drop rows paged in for the old one.
  useEffect(() => {
    if (lastTaskRef.current !== taskId) {
      lastTaskRef.current = taskId
      setEarlier([])
      setOlderAvailable(null)
      setPinned(true)
    }
  }, [taskId])

  const live = data?.entries ?? []
  const moreEarlier = olderAvailable ?? Boolean(data?.has_earlier)
  const rows = useMemo(() => dedupe([...earlier, ...live]), [earlier, live])
  const minId = rows[0]?.id ?? null
  const atCap = rows.length >= MAX_ROWS_HELD
  // Follow the newest line while the reader is at the bottom (or on expand).
  const tailKey = rows.length > 0 ? `${rows.length}:${rows[rows.length - 1].id}` : '0'

  useEffect(() => {
    if (open && pinned && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [open, pinned, tailKey])

  const onScroll = useCallback(() => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 48)
  }, [])

  const toggle = useCallback(() => {
    const next = !readOpen()

    persistOpen(next)
    setOpen(next)
  }, [])

  const liveCount = live.length

  const loadEarlier = useCallback(() => {
    if (!minId || loadingEarlier || !moreEarlier || atCap) {
      return
    }

    setLoadingEarlier(true)

    fetchOutput(board, taskId, minId)
      .then(page => {
        setEarlier(prev => {
          // Grow only up to the cap; keep the newest of the paged-in rows
          // (the ones nearest what is already on screen).
          const room = Math.max(0, MAX_ROWS_HELD - liveCount - prev.length)

          return room > 0 ? dedupe([...page.entries.slice(-room), ...prev]) : prev
        })
        setOlderAvailable(page.has_earlier)
        setPinned(false)
      })
      .catch((err: unknown) => host.notify({ kind: 'error', message: err instanceof Error ? err.message : String(err) }))
      .finally(() => setLoadingEarlier(false))
  }, [atCap, board, liveCount, loadingEarlier, minId, moreEarlier, taskId])

  const summary = stateSummary(data, error, running, status)
  const canLoadEarlier = canPageBack(data?.state, moreEarlier, atCap, minId)

  return (
    <section className="flex flex-col gap-2" data-slot="kanban-output-log" data-open={open}>
      <button
        aria-expanded={open}
        className="flex w-full items-center gap-2 text-left text-[0.75rem] font-medium text-(--ui-text-secondary) transition-colors hover:text-(--ui-text-primary)"
        onClick={toggle}
        type="button"
      >
        <Codicon className={cn('transition-transform', open && 'rotate-90')} name="chevron-right" size="0.8rem" />
        <Codicon name="output" size="0.8rem" />
        <span>Output log</span>
        {running && isFetching && <Loader type="lemniscate-bloom" />}
        <span className="ml-auto min-w-0 truncate text-[0.6875rem] font-normal text-(--ui-text-quaternary)">
          {summary}
        </span>
      </button>

      {open && (
        <OutputLogPanel
          atCap={atCap}
          canLoadEarlier={canLoadEarlier}
          data={data}
          error={error}
          loadingEarlier={loadingEarlier}
          moreEarlier={moreEarlier}
          onLoadEarlier={loadEarlier}
          onScroll={onScroll}
          rows={rows}
          scrollRef={scrollRef}
        />
      )}
    </section>
  )
}
