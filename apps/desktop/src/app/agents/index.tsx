import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'

import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { Codicon } from '@/components/ui/codicon'
import { FadeText } from '@/components/ui/fade-text'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { compactNumber } from '@/lib/format'
import { AlertCircle, CheckCircle2 } from '@/lib/icons'
import { type TodoItem, todoTree } from '@/lib/todos'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import {
  $backgroundStatusBySession,
  allBackgroundProcesses,
  BACKGROUND_POLL_MS,
  type BackgroundProcessOverviewItem,
  backgroundProcessPollingSessionIds,
  dismissBackgroundProcess,
  refreshBackgroundProcesses,
  stopBackgroundProcess
} from '@/store/composer-status'
import { $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import {
  $sessionTodoOverviewRows,
  cancelTodoOverviewItem,
  dismissSessionTodoOverviewRow,
  dismissTodoOverviewItem,
  moveTodoOverviewItem,
  type SessionTodoOverviewRow
} from '@/store/session-todos-overview'
import {
  $subagentsBySession,
  allSubagents,
  buildSubagentTree,
  type SubagentNode,
  type SubagentStatus,
  type SubagentStreamEntry
} from '@/store/subagents'

import { Panel, PanelEmpty, PanelHeader, PanelSectionLabel } from '../overlays/panel'

// Mirrors statusGlyph() in tool-fallback.tsx so subagent rows speak the
// same visual vocabulary as the chat tool blocks.
function statusGlyph(status: SubagentStatus, a: Translations['agents']): ReactNode {
  if (status === 'running' || status === 'queued') {
    return (
      <GlyphSpinner
        ariaLabel={a.running}
        className="size-3.5 shrink-0 text-[0.95rem] text-muted-foreground/80"
        spinner="breathe"
      />
    )
  }

  if (status === 'failed' || status === 'interrupted') {
    return <AlertCircle aria-label={a.failed} className="size-3.5 shrink-0 text-destructive" />
  }

  return <CheckCircle2 aria-label={a.done} className="size-3.5 shrink-0 text-emerald-600/85 dark:text-emerald-400/85" />
}

const STREAM_TONE: Record<SubagentStreamEntry['kind'], string> = {
  progress: 'text-muted-foreground/75',
  summary: 'text-foreground/85',
  thinking: 'text-muted-foreground/80',
  tool: 'text-foreground/85'
}

function streamGlyph(entry: SubagentStreamEntry): ReactNode {
  if (entry.isError) {
    return <AlertCircle aria-hidden className="mt-0.5 size-3 shrink-0 text-destructive" />
  }

  if (entry.kind === 'tool') {
    return <span aria-hidden className="mt-0.5 size-1.5 shrink-0 rounded-full bg-foreground/55" />
  }

  if (entry.kind === 'summary') {
    return <CheckCircle2 aria-hidden className="mt-0.5 size-3 shrink-0 text-emerald-600/85 dark:text-emerald-400/85" />
  }

  if (entry.kind === 'thinking') {
    return (
      <span aria-hidden className="font-mono text-[0.7rem] leading-none text-muted-foreground/70">
        …
      </span>
    )
  }

  return <span aria-hidden className="mt-0.5 size-1 shrink-0 rounded-full bg-muted-foreground/55" />
}

interface AgentsViewProps {
  onClose: () => void
}

export function AgentsView({ onClose }: AgentsViewProps) {
  const { t } = useI18n()

  return (
    <Panel closeLabel={t.agents.close} onClose={onClose}>
      <AgentsPanelContent />
    </Panel>
  )
}

// The bare content, with no overlay chrome of its own — the standing
// right-sidebar panel contribution (registered in app/contrib/controller.tsx
// as `agents`) renders this directly inside the pane-shell zone, which
// already supplies the tab strip / close button. AgentsView above keeps the
// legacy full-screen overlay working (route /agents) by wrapping the same
// content in `Panel`.
export function AgentsPanelContent() {
  const { t } = useI18n()
  const subagentsBySession = useStore($subagentsBySession)
  const backgroundStatusBySession = useStore($backgroundStatusBySession)
  const sessionStates = useStore($sessionStates)
  const sessions = useStore($sessions)

  // Aggregate every session, matching the status-bar indicator — a subagent
  // running in a background session must still be visible here, or the two
  // desync ("Agents N running" vs an empty tree).
  const tree = useMemo(() => buildSubagentTree(allSubagents(subagentsBySession)), [subagentsBySession])

  const backgroundProcesses = useMemo(
    () => allBackgroundProcesses(backgroundStatusBySession, sessionStates, sessions),
    [backgroundStatusBySession, sessionStates, sessions]
  )

  // Poll every active/recently-active session's background processes while
  // this panel is mounted — the composer's own poll (status-stack/index.tsx)
  // only covers whichever ONE session tab is currently focused, so a
  // background process started in a session the user isn't looking at would
  // never surface here without this. Bounded to running/recent sessions
  // (composer-status.ts::backgroundProcessPollingSessionIds), not every
  // session ever seen, to avoid hammering the gateway.
  useEffect(() => {
    const poll = () => {
      for (const runtimeId of backgroundProcessPollingSessionIds($backgroundStatusBySession.get(), $sessionStates.get(), $sessions.get())) {
        void refreshBackgroundProcesses(runtimeId)
      }
    }

    poll()
    const timer = window.setInterval(poll, BACKGROUND_POLL_MS)

    return () => window.clearInterval(timer)
  }, [])

  // Task overview sits ABOVE the subagent tree (user-requested order): it is
  // the standing, always-relevant summary (what's left to do), while the
  // subagent tree is transient activity that comes and goes as delegations
  // run. NO overscroll-behavior on this inner wrapper: Chromium registers any
  // `overflow-y-auto` element as its own scroll container the moment content
  // could theoretically overflow it, and `overscroll-behavior: contain`
  // stops scroll-chaining to a parent EVEN WHEN THIS ELEMENT HAS NO SCROLL
  // ROOM OF ITS OWN (scrollHeight === clientHeight, which is exactly what
  // happens here — flex-1 lets it grow to fit its content instead of ever
  // clipping). The real scrollable surface is the OUTER pane layer
  // (`absolute inset-0 overflow-auto` in tree-group.tsx), so once contain
  // ate the wheel event here, it never reached that outer scroller and the
  // whole panel felt unscrollable with a mouse wheel. Reproduced with a
  // synthetic CDP wheel dispatch and a minimal repro outside React before
  // this fix; confirmed fixed by simply dropping overscroll-behavior here.
  if (tree.length === 0) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto pl-3">
        <TaskOverviewSection />
        <BackgroundProcessesSection processes={backgroundProcesses} />
        <PanelEmpty description={t.agents.emptyDesc} icon="hubot" title={t.agents.emptyTitle} />
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto pl-3">
      <TaskOverviewSection />
      <BackgroundProcessesSection processes={backgroundProcesses} />
      <div className="flex min-h-0 flex-1 flex-col gap-4">
        <PanelHeader subtitle={t.agents.subtitle} title={t.agents.title} />
        <SubagentTree tree={tree} />
      </div>
    </div>
  )
}

const fmtDuration = (seconds: number | undefined, a: Translations['agents']) => {
  if (!seconds || seconds <= 0) {
    return ''
  }

  if (seconds < 60) {
    return a.durationSeconds(seconds.toFixed(1))
  }

  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)

  return a.durationMinutes(m, s)
}

const fmtTokens = (value: number | undefined, a: Translations['agents']) =>
  value ? a.tokens(compactNumber(value)) : ''

// Distinct contract from coarseElapsed: rounds to the second (this ticks live),
// and hours are unbounded ("25h", never "1d"). Kept local on purpose.
const fmtAge = (updatedAt: number, nowMs: number, a: Translations['agents']) => {
  const s = Math.max(0, Math.round((nowMs - updatedAt) / 1000))

  if (s < 2) {
    return a.ageNow
  }

  if (s < 60) {
    return a.ageSeconds(s)
  }

  const m = Math.floor(s / 60)

  return m < 60 ? a.ageMinutes(m) : a.ageHours(Math.floor(m / 60))
}

const flatten = (nodes: readonly SubagentNode[]): SubagentNode[] =>
  nodes.flatMap(node => [node, ...flatten(node.children)])

interface RootGroup {
  id: string
  delegationIndex: number
  nodes: SubagentNode[]
  taskCount: number
}

function groupDelegations(roots: readonly SubagentNode[]): RootGroup[] {
  const groups: RootGroup[] = []
  let n = 0

  for (const node of roots) {
    // Exact grouping when the backend tags workers with their batch id —
    // concurrent or nested fan-outs of the same shape must not merge.
    if (node.delegationId) {
      const byId = groups.find(g => g.id === `delegation:${node.delegationId}`)

      if (byId) {
        byId.nodes.push(node)

        continue
      }

      n += 1
      groups.push({
        id: `delegation:${node.delegationId}`,
        delegationIndex: n,
        nodes: [node],
        taskCount: node.taskCount
      })

      continue
    }

    // Older backends (no delegation_id): heuristic grouping by shape + time.
    const prev = groups.at(-1)
    const prevTail = prev?.nodes.at(-1)
    const closeInTime = prevTail ? Math.abs(node.startedAt - prevTail.startedAt) <= 5_000 : false

    const sameShape =
      prev && !prev.id.startsWith('delegation:') && node.taskCount > 1 && prev.taskCount === node.taskCount

    const uniqueStep = prev ? !prev.nodes.some(item => item.taskIndex === node.taskIndex) : false

    if (prev && sameShape && closeInTime && uniqueStep) {
      prev.nodes.push(node)

      continue
    }

    if (node.taskCount > 1) {
      n += 1
      groups.push({ id: `delegation-${n}`, delegationIndex: n, nodes: [node], taskCount: node.taskCount })

      continue
    }

    groups.push({ id: node.id, delegationIndex: 0, nodes: [node], taskCount: node.taskCount })
  }

  return groups
}

function SubagentTree({ tree }: { tree: SubagentNode[] }) {
  const { t } = useI18n()
  const flat = useMemo(() => flatten(tree), [tree])
  const groups = useMemo(() => groupDelegations(tree), [tree])
  const [nowMs, setNowMs] = useState(() => Date.now())

  const active = flat.filter(n => n.status === 'running' || n.status === 'queued').length
  const failed = flat.filter(n => n.status === 'failed' || n.status === 'interrupted').length
  const tools = flat.reduce((sum, n) => sum + (n.toolCount ?? 0), 0)
  const files = flat.reduce((sum, n) => sum + n.filesRead.length + n.filesWritten.length, 0)
  const tokens = flat.reduce((sum, n) => sum + (n.inputTokens ?? 0) + (n.outputTokens ?? 0), 0)
  const cost = flat.reduce((sum, n) => sum + (n.costUsd ?? 0), 0)

  const visible = usePaneVisible()

  useEffect(() => {
    if (active <= 0 || !visible || typeof window === 'undefined') {
      return
    }

    const id = window.setInterval(() => setNowMs(Date.now()), 500)

    return () => window.clearInterval(id)
  }, [active, visible])

  if (tree.length === 0) {
    return (
      <div className="grid place-items-center gap-3 py-12 text-center">
        <Codicon className="text-muted-foreground/60" name="hubot" size="1.5rem" />
        <p className="text-sm font-medium text-foreground/90">{t.agents.emptyTitle}</p>
        <p className="max-w-md text-xs leading-relaxed text-muted-foreground/75">{t.agents.emptyDesc}</p>
      </div>
    )
  }

  const summary = [
    t.agents.agentsCount(flat.length),
    active > 0 ? t.agents.activeCount(active) : '',
    failed > 0 ? t.agents.failedCount(failed) : '',
    tools > 0 ? t.agents.toolsCount(tools) : '',
    files > 0 ? t.agents.filesCount(files) : '',
    tokens > 0 ? fmtTokens(tokens, t.agents) : '',
    cost > 0 ? `$${cost.toFixed(2)}` : ''
  ].filter(Boolean)

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-hidden">
      <p className="shrink-0 text-[0.7rem] text-muted-foreground/70">{summary.join(' · ')}</p>
      <div className="min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto pr-1">
        <div className="flex min-w-0 flex-col gap-6">
          {groups.map(group => (
            <DelegationGroup group={group} key={group.id} nowMs={nowMs} />
          ))}
        </div>
      </div>
    </div>
  )
}

function DelegationGroup({ group, nowMs }: { group: RootGroup; nowMs: number }) {
  const { t } = useI18n()

  if (group.nodes.length === 1 && group.taskCount <= 1) {
    return <SubagentRow node={group.nodes[0]!} nowMs={nowMs} />
  }

  const activeWorkers = group.nodes.filter(n => n.status === 'running' || n.status === 'queued').length

  return (
    <section className="grid min-w-0 gap-3">
      <p className="text-[0.66rem] font-medium uppercase tracking-wider text-muted-foreground/70">
        {group.delegationIndex > 0 ? t.agents.delegation(group.delegationIndex) : ''}{' '}
        <span className="text-muted-foreground/50">·</span> {t.agents.workers(group.nodes.length)}
        {activeWorkers > 0 ? <span className="text-primary/85"> · {t.agents.workersActive(activeWorkers)}</span> : null}
      </p>
      <div className="grid min-w-0 gap-4">
        {group.nodes.map(node => (
          <SubagentRow key={node.id} node={node} nowMs={nowMs} />
        ))}
      </div>
    </section>
  )
}

function StreamLine({
  active,
  entry,
  parentRunning,
  rowKey
}: {
  active: boolean
  entry: SubagentStreamEntry
  parentRunning: boolean
  rowKey: string
}) {
  const { t } = useI18n()
  const enterRef = useEnterAnimation(parentRunning, `subagent-stream:${rowKey}`)
  const isMono = entry.kind === 'tool'
  const tone = entry.isError ? 'text-destructive' : STREAM_TONE[entry.kind]

  return (
    <div className="flex min-w-0 items-baseline gap-2 text-[0.72rem] leading-relaxed" ref={enterRef}>
      <span className="flex h-[0.95rem] shrink-0 items-center">{streamGlyph(entry)}</span>
      <span className={cn('min-w-0 flex-1 wrap-anywhere', tone, isMono && 'font-mono text-[0.69rem]')}>
        {entry.text}
        {active ? (
          <GlyphSpinner
            ariaLabel={t.agents.streaming}
            className="ml-1 inline-block size-2.5 align-middle text-muted-foreground/70"
            spinner="breathe"
          />
        ) : null}
      </span>
    </div>
  )
}

export function SubagentRow({ node, depth = 0, nowMs }: { node: SubagentNode; depth?: number; nowMs: number }) {
  const { t } = useI18n()
  const running = node.status === 'running' || node.status === 'queued'
  const elapsed = useElapsedSeconds(running, `subagent:${node.id}`, node.startedAt)

  const durationSeconds =
    typeof node.durationSeconds === 'number' ? Math.max(0, Math.round(node.durationSeconds)) : elapsed

  const [open, setOpen] = useState(() => running || depth < 2)
  const enterRef = useEnterAnimation(true, `subagent-row:${node.id}`)

  useEffect(() => {
    if (running) {
      setOpen(true)
    }
  }, [running])

  const visibleRows = open ? node.stream.slice(-10) : node.stream.slice(-2)
  const fileLines = [...node.filesWritten.map(p => `+ ${p}`), ...node.filesRead.map(p => `· ${p}`)]

  const subtitle = [
    node.model,
    fmtDuration(durationSeconds, t.agents),
    node.toolCount ? t.agents.toolsCount(node.toolCount) : '',
    fmtTokens((node.inputTokens ?? 0) + (node.outputTokens ?? 0), t.agents),
    t.agents.updatedAgo(fmtAge(node.updatedAt, nowMs, t.agents))
  ].filter(Boolean)

  return (
    <div className={cn('grid min-w-0 max-w-full gap-2', depth > 0 && 'pl-4')} data-slot="tool-block" ref={enterRef}>
      <button
        aria-expanded={open}
        className="group flex w-full min-w-0 items-start gap-2.5 text-left"
        onClick={() => setOpen(v => !v)}
        type="button"
      >
        <span className="mt-0.5 flex h-[1.1rem] shrink-0 items-center">{statusGlyph(node.status, t.agents)}</span>
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span
            className={cn(
              'wrap-anywhere text-[0.82rem] font-medium leading-[1.1rem] text-foreground/90 transition-colors group-hover:text-foreground',
              running && 'shimmer text-foreground/65'
            )}
          >
            {node.goal}
          </span>
          {subtitle.length > 0 ? (
            <FadeText className="text-[0.66rem] leading-[1.05rem] text-muted-foreground/65">
              {subtitle.join(' · ')}
            </FadeText>
          ) : null}
        </span>
        {running ? <ActivityTimerText className="mt-1 shrink-0 text-[0.6rem]" seconds={durationSeconds} /> : null}
      </button>

      {visibleRows.length > 0 ? (
        <div className="grid min-w-0 gap-1 pl-6" data-selectable-text="true">
          {visibleRows.map((entry, i) => (
            <StreamLine
              active={running && i === visibleRows.length - 1}
              entry={entry}
              key={`${entry.kind}:${entry.at}:${i}`}
              parentRunning={running}
              rowKey={`${node.id}:${entry.kind}:${entry.at}`}
            />
          ))}
        </div>
      ) : null}

      {open && fileLines.length > 0 ? (
        <div className="grid min-w-0 gap-0.5 pl-6" data-selectable-text="true">
          <p className="text-[0.58rem] font-medium tracking-wider text-muted-foreground/60 uppercase">
            {t.agents.files}
          </p>
          {fileLines.slice(0, 8).map(line => (
            <p className="wrap-break-word font-mono text-[0.67rem] leading-relaxed text-muted-foreground/80" key={line}>
              {line}
            </p>
          ))}
          {fileLines.length > 8 ? (
            <p className="font-mono text-[0.67rem] leading-relaxed text-muted-foreground/65">
              {t.agents.moreFiles(fileLines.length - 8)}
            </p>
          ) : null}
        </div>
      ) : null}

      {node.children.length > 0 ? (
        <div className="grid min-w-0 gap-3 pl-6">
          {node.children.map(child => (
            <SubagentRow depth={depth + 1} key={child.id} node={child} nowMs={nowMs} />
          ))}
        </div>
      ) : null}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Task overview — todo lists across every session this window has observed.
// Distinct data source from the subagent tree above: reads
// store/session-todos-overview.ts (all-sessions, persists for the window's
// lifetime) rather than store/subagents.ts (current-turn subagents only).
// ---------------------------------------------------------------------------

const TASK_ROWS_COLLAPSED_LIMIT = 3

// Drag-and-drop payload for moving one todo item between session rows. Plain
// HTML5 DnD (not the pointer-drag session machinery in session-drag.ts —
// that's built for session TILES with stack/split/link targeting across pane
// zones; this is a much smaller in-place reorder confined to one list inside
// one sidebar section, so the browser's native drag events are simpler and
// sufficient). MIME-typed so a drop handler can reject anything that isn't
// one of our own items before touching dataTransfer's JSON.
const TODO_DRAG_MIME = 'application/x-hermes-todo-item'

interface TodoDragPayload {
  fromStoredSessionId: string
  itemId: string
}

function TaskOverviewRow({ row }: { row: SessionTodoOverviewRow }) {
  const { t } = useI18n()
  const [showDone, setShowDone] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const ordered = useMemo(() => todoTree(row.todos), [row.todos])
  const counted = row.todos.filter(item => item.status !== 'cancelled')
  const doneCount = counted.filter(item => item.status === 'completed').length
  const activeEntries = ordered.filter(([item]) => item.status !== 'completed' && item.status !== 'cancelled')
  const doneEntries = ordered.filter(([item]) => item.status === 'completed')
  const visibleDone = showDone ? doneEntries : []

  const dismiss = () => {
    dismissSessionTodoOverviewRow(row.storedSessionId)
  }

  const dismissItem = (item: TodoItem) => {
    if (item.status === 'pending' || item.status === 'in_progress') {
      void cancelTodoOverviewItem(row.storedSessionId, item.id)
    } else {
      dismissTodoOverviewItem(row.storedSessionId, item.id)
    }
  }

  const dragTypes = (e: { dataTransfer: DataTransfer }) => Array.from(e.dataTransfer.types)

  return (
    <div
      className={cn(
        'group/task-row grid min-w-0 gap-2 rounded-md p-1.5 transition-colors hover:bg-(--ui-row-hover-background)',
        dragOver && 'bg-(--ui-row-hover-background) ring-1 ring-primary/60'
      )}
      onDragLeave={() => setDragOver(false)}
      onDragOver={e => {
        if (!dragTypes(e).includes(TODO_DRAG_MIME)) {
          return
        }

        e.preventDefault()
        e.dataTransfer.dropEffect = 'move'
        setDragOver(true)
      }}
      onDrop={e => {
        if (!dragTypes(e).includes(TODO_DRAG_MIME)) {
          return
        }

        e.preventDefault()
        setDragOver(false)

        let payload: TodoDragPayload | null = null

        try {
          payload = JSON.parse(e.dataTransfer.getData(TODO_DRAG_MIME)) as TodoDragPayload
        } catch {
          return
        }

        if (payload && payload.fromStoredSessionId !== row.storedSessionId) {
          void moveTodoOverviewItem(payload.fromStoredSessionId, row.storedSessionId, payload.itemId)
        }
      }}
    >
      <div className="flex min-w-0 items-start justify-between gap-2">
        <p className="min-w-0 flex-1 truncate text-[0.78rem] font-medium text-foreground/90">{row.title}</p>
        <div className="flex shrink-0 items-center gap-1.5">
          <span className="text-[0.66rem] tabular-nums text-muted-foreground/60">
            {t.agents.taskOverviewProgress(doneCount, counted.length)}
          </span>
          <Tip label={t.agents.taskOverviewDismiss}>
            <button
              aria-label={t.agents.taskOverviewDismiss}
              className="grid size-5 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 transition hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/task-row:opacity-100"
              onClick={dismiss}
              type="button"
            >
              <Codicon name="close" size="0.7rem" />
            </button>
          </Tip>
        </div>
      </div>
      <div className="grid min-w-0 gap-1 pl-1">
        {activeEntries.map(([item, depth]) => (
          <TaskOverviewItemRow
            depth={depth}
            item={item}
            key={item.id}
            onDismiss={() => dismissItem(item)}
            storedSessionId={row.storedSessionId}
          />
        ))}
        {visibleDone.map(([item, depth]) => (
          <TaskOverviewItemRow
            depth={depth}
            item={item}
            key={item.id}
            onDismiss={() => dismissItem(item)}
            storedSessionId={row.storedSessionId}
          />
        ))}
      </div>
      {doneEntries.length > 0 ? (
        <button
          className="w-fit pl-1 text-[0.66rem] text-muted-foreground/60 hover:text-foreground hover:underline"
          onClick={() => setShowDone(v => !v)}
          type="button"
        >
          {showDone ? t.agents.taskOverviewHideDone : t.agents.taskOverviewShowDone(doneEntries.length)}
        </button>
      ) : null}
    </div>
  )
}

function TaskOverviewItemRow({
  depth,
  item,
  onDismiss,
  storedSessionId
}: {
  depth: number
  item: TodoItem
  onDismiss: () => void
  storedSessionId: string
}) {
  const { t } = useI18n()
  const done = item.status === 'completed'
  const active = item.status === 'in_progress'
  const cancelled = item.status === 'cancelled'
  const canCancel = item.status === 'pending' || item.status === 'in_progress'
  const dismissLabel = canCancel ? t.agents.taskOverviewCancel : t.agents.taskOverviewDismiss

  return (
    <div
      className={cn('group/task-item flex min-w-0 items-start gap-1.5', depth > 0 && 'pl-4')}
      draggable
      onDragStart={e => {
        const payload: TodoDragPayload = { fromStoredSessionId: storedSessionId, itemId: item.id }
        e.dataTransfer.effectAllowed = 'move'
        e.dataTransfer.setData(TODO_DRAG_MIME, JSON.stringify(payload))
        // A plain-text fallback so dropping on something outside our own
        // targets (e.g. a text field) pastes something legible instead of
        // nothing — never load-bearing for the actual move.
        e.dataTransfer.setData('text/plain', item.content)
      }}
    >
      <span
        aria-hidden
        className={cn(
          'mt-1 size-1.5 shrink-0 rounded-full',
          done ? 'bg-emerald-500/70' : active ? 'bg-primary' : 'bg-muted-foreground/40'
        )}
      />
      <span
        className={cn(
          'min-w-0 flex-1 text-[0.72rem] leading-[1.15rem] wrap-anywhere',
          done || cancelled ? 'text-muted-foreground/55 line-through' : 'text-foreground/85'
        )}
      >
        {item.content}
        {cancelled ? (
          <span className="ml-1.5 text-[0.62rem] text-muted-foreground/50 no-underline">
            {t.agents.taskOverviewCancelled}
          </span>
        ) : null}
      </span>
      <Tip label={dismissLabel}>
        <button
          aria-label={dismissLabel}
          className="grid size-4 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 transition hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/task-item:opacity-100"
          onClick={onDismiss}
          type="button"
        >
          <Codicon name="close" size="0.6rem" />
        </button>
      </Tip>
    </div>
  )
}

function TaskOverviewSection() {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)
  // rows lives directly in an atom (see store/session-todos-overview.ts) —
  // useStore subscribes to the actual payload, not an indirect counter a
  // consumer must remember to re-derive data through. That indirection
  // (a separate "tick" atom + a plain function call to fetch the real data)
  // was reproduced live as the exact cause of the "todo list doesn't show up
  // until I close and reopen the sidebar" bug: the tick changed and the
  // listener fired, but React never repainted the already-mounted panel with
  // the fresh data.
  const rows = useStore($sessionTodoOverviewRows)
  const shown = expanded ? rows : rows.slice(0, TASK_ROWS_COLLAPSED_LIMIT)
  const hiddenCount = rows.length - shown.length

  return (
    <div className="grid min-h-0 shrink-0 gap-2 border-t border-(--ui-border-subtle) pt-3">
      <PanelSectionLabel>{t.agents.taskOverviewTitle}</PanelSectionLabel>
      {rows.length === 0 ? (
        <p className="px-1 text-[0.7rem] leading-relaxed text-muted-foreground/60">{t.agents.taskOverviewEmptyDesc}</p>
      ) : (
        <div className="grid min-w-0 gap-1 overflow-y-auto">
          {shown.map(row => (
            <TaskOverviewRow key={row.storedSessionId} row={row} />
          ))}
          {hiddenCount > 0 ? (
            <button
              className="w-fit px-1.5 text-[0.68rem] text-muted-foreground/60 hover:text-foreground hover:underline"
              onClick={() => setExpanded(true)}
              type="button"
            >
              +{hiddenCount}
            </button>
          ) : null}
          </div>
          )}
          </div>
          )
          }

          // ---------------------------------------------------------------------------
          // Background processes — terminal(background=True) work across every session
          // this window has observed, aggregated the same way the subagent tree
          // aggregates $subagentsBySession (store/composer-status.ts::allBackgroundProcesses).
          // Previously this data only ever reached the composer's own status stack for
          // whichever ONE session tab was focused; a background process running in a
          // session the user wasn't looking at was invisible everywhere else.
          // ---------------------------------------------------------------------------

          function backgroundStatusGlyph(state: BackgroundProcessOverviewItem['state'], a: Translations['agents']): ReactNode {
          if (state === 'running') {
          return (
          <GlyphSpinner
            ariaLabel={a.running}
            className="size-3.5 shrink-0 text-[0.95rem] text-muted-foreground/80"
            spinner="breathe"
          />
          )
          }

          if (state === 'failed') {
          return <AlertCircle aria-label={a.failed} className="size-3.5 shrink-0 text-destructive" />
          }

          return <CheckCircle2 aria-label={a.done} className="size-3.5 shrink-0 text-emerald-600/85 dark:text-emerald-400/85" />
          }

          function BackgroundProcessRow({ item }: { item: BackgroundProcessOverviewItem }) {
          const { t } = useI18n()
          const running = item.state === 'running'
          const actionLabel = running ? t.agents.backgroundStop : t.agents.backgroundDismiss

          const onAction = () => {
          if (running) {
          void stopBackgroundProcess(item.runtimeSessionId, item.id)
          } else {
          dismissBackgroundProcess(item.runtimeSessionId, item.id)
          }
          }

          return (
          <div className="group/bg-item flex min-w-0 items-start gap-1.5 rounded-md p-1 hover:bg-(--ui-row-hover-background)">
          <span className="mt-0.5 flex h-[0.95rem] shrink-0 items-center">{backgroundStatusGlyph(item.state, t.agents)}</span>
          <span
            className={cn(
              'min-w-0 flex-1 text-[0.72rem] leading-[1.15rem] wrap-anywhere',
              running ? 'text-foreground/85' : 'text-muted-foreground/65'
            )}
          >
            {item.title}
            {item.exitCode ? <span className="ml-1.5 text-[0.62rem] text-destructive/80">exit {item.exitCode}</span> : null}
          </span>
          <Tip label={actionLabel}>
            <button
              aria-label={actionLabel}
              className="grid size-4 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 transition hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/bg-item:opacity-100"
              onClick={onAction}
              type="button"
            >
              <Codicon name="close" size="0.6rem" />
            </button>
          </Tip>
          </div>
          )
          }

          function BackgroundProcessesSection({ processes }: { processes: BackgroundProcessOverviewItem[] }) {
          const { t } = useI18n()

          if (processes.length === 0) {
          return null
          }

          return (
          <div className="grid min-h-0 shrink-0 gap-2 border-t border-(--ui-border-subtle) pt-3">
          <PanelSectionLabel>{t.agents.backgroundTitle}</PanelSectionLabel>
          <div className="grid min-w-0 gap-1">
            {processes.map(item => (
              <BackgroundProcessRow item={item} key={`${item.runtimeSessionId}:${item.id}`} />
            ))}
          </div>
          </div>
          )
          }
