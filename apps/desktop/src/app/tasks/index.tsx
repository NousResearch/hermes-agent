import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { $subagentsBySession, type SubagentNode } from '@/store/subagents'
import { $todosBySession } from '@/store/todos'

import { PAGE_INSET_X } from '../layout-constants'
import { PageSearchShell } from '../page-search-shell'
import { sessionRoute } from '../routes'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { aggregateLiveTaskSessions, type LiveTaskSession, type LiveTodoStatus } from './aggregation'

function includesQuery(value: null | string | undefined, query: string): boolean {
  return (value || '').toLowerCase().includes(query)
}

function taskStatusTone(status: LiveTodoStatus): string {
  switch (status) {
    case 'completed':
      return 'bg-emerald-500/12 text-emerald-300 border-emerald-500/20'

    case 'cancelled':
      return 'bg-slate-500/12 text-slate-300 border-slate-500/20'

    case 'in_progress':
      return 'bg-sky-500/12 text-sky-300 border-sky-500/20'

    case 'pending':

    default:
      return 'bg-amber-500/12 text-amber-200 border-amber-500/20'
  }
}

function taskStatusLabel(status: LiveTodoStatus): string {
  switch (status) {
    case 'completed':
      return 'Done'

    case 'cancelled':
      return 'Cancelled'

    case 'in_progress':
      return 'In progress'

    case 'pending':

    default:
      return 'Pending'
  }
}

function subagentStatusTone(status: SubagentNode['status']): string {
  switch (status) {
    case 'completed':
      return 'bg-emerald-500/12 text-emerald-300 border-emerald-500/20'

    case 'failed':

    case 'interrupted':
      return 'bg-rose-500/12 text-rose-300 border-rose-500/20'

    case 'queued':
      return 'bg-violet-500/12 text-violet-200 border-violet-500/20'

    case 'running':

    default:
      return 'bg-sky-500/12 text-sky-300 border-sky-500/20'
  }
}

function subagentStatusLabel(status: SubagentNode['status']): string {
  switch (status) {
    case 'completed':
      return 'Completed'

    case 'failed':
      return 'Failed'

    case 'interrupted':
      return 'Interrupted'

    case 'queued':
      return 'Queued'

    case 'running':

    default:
      return 'Running'
  }
}

function latestSubagentLine(node: SubagentNode): null | string {
  return node.stream.at(-1)?.text?.trim() || node.summary?.trim() || null
}

function matchesSession(entry: LiveTaskSession, query: string): boolean {
  if (!query) {
    return true
  }

  if (includesQuery(entry.title, query) || includesQuery(entry.id, query) || includesQuery(entry.session?.cwd, query)) {
    return true
  }

  if (entry.todos.some(todo => includesQuery(todo.content, query) || includesQuery(todo.status, query))) {
    return true
  }

  const stack = [...entry.subagents]

  while (stack.length > 0) {
    const node = stack.pop()!

    if (
      includesQuery(node.goal, query) ||
      includesQuery(node.currentTool, query) ||
      includesQuery(node.summary, query) ||
      includesQuery(latestSubagentLine(node), query)
    ) {
      return true
    }

    stack.push(...node.children)
  }

  return false
}

interface TasksViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function TasksView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: TasksViewProps) {
  const navigate = useNavigate()
  const sessions = useStore($sessions)
  const sessionStates = useStore($sessionStates)
  const todosBySession = useStore($todosBySession)
  const subagentsBySession = useStore($subagentsBySession)
  const [query, setQuery] = useState('')

  const liveSessions = useMemo(
    () => aggregateLiveTaskSessions({ sessions, sessionStates, subagentsBySession, todosBySession }),
    [sessions, sessionStates, subagentsBySession, todosBySession]
  )

  const normalizedQuery = query.trim().toLowerCase()

  const visibleSessions = useMemo(
    () => liveSessions.filter(entry => matchesSession(entry, normalizedQuery)),
    [liveSessions, normalizedQuery]
  )

  const totals = useMemo(() => {
    let todoCount = 0
    let subagentCount = 0
    let blockedCount = 0
    let busyCount = 0

    for (const entry of liveSessions) {
      todoCount += entry.activeTodoCount
      subagentCount += entry.activeSubagentCount
      blockedCount += Number(entry.needsInput)
      busyCount += Number(entry.busy)
    }

    return {
      sessions: liveSessions.length,
      todos: todoCount,
      subagents: subagentCount,
      blocked: blockedCount,
      busy: busyCount
    }
  }, [liveSessions])

  return (
    <PageSearchShell
      {...props}
      activeTab="live"
      filters={
        <>
          <SummaryPill label="Live sessions" value={totals.sessions} />
          <SummaryPill label="Busy now" value={totals.busy} />
          <SummaryPill label="Todo items" value={totals.todos} />
          <SummaryPill label="Subagents" value={totals.subagents} />
          <SummaryPill label="Needs input" value={totals.blocked} />
        </>
      }
      onSearchChange={setQuery}
      searchHidden={liveSessions.length === 0}
      searchPlaceholder="Search live tasks…"
      searchValue={query}
      tabs={[{ id: 'live', label: 'Live tasks' }]}
    >
      <div className={cn('h-full overflow-y-auto py-3', PAGE_INSET_X)}>
        {visibleSessions.length === 0 ? (
          <EmptyState hasQuery={normalizedQuery.length > 0} />
        ) : (
          <div className="space-y-3 pb-6">
            {visibleSessions.map(entry => (
              <section
                className="rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) p-4 shadow-sm"
                key={entry.id}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="truncate text-sm font-semibold text-foreground">{entry.title}</h2>
                      {entry.busy && <Pill className="bg-sky-500/12 text-sky-300 border-sky-500/20">Running</Pill>}
                      {entry.needsInput && (
                        <Pill className="bg-amber-500/12 text-amber-200 border-amber-500/20">Needs input</Pill>
                      )}
                      {entry.failedSubagentCount > 0 && (
                        <Pill className="bg-rose-500/12 text-rose-300 border-rose-500/20">
                          {entry.failedSubagentCount} issue{entry.failedSubagentCount === 1 ? '' : 's'}
                        </Pill>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2 text-xs text-(--ui-text-tertiary)">
                      <span>{entry.activeTodoCount} active todo{entry.activeTodoCount === 1 ? '' : 's'}</span>
                      <span>•</span>
                      <span>{entry.activeSubagentCount} active subagent{entry.activeSubagentCount === 1 ? '' : 's'}</span>
                      {entry.session?.cwd ? (
                        <>
                          <span>•</span>
                          <span className="max-w-[48rem] truncate">{entry.session.cwd}</span>
                        </>
                      ) : null}
                    </div>
                  </div>

                  <Button
                    disabled={!entry.routeId}
                    onClick={() => entry.routeId && navigate(sessionRoute(entry.routeId))}
                    size="sm"
                    type="button"
                    variant="outline"
                  >
                    Open session
                  </Button>
                </div>

                <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
                  <div className="space-y-2">
                    <SectionTitle title="Todo list" />
                    {entry.todos.length === 0 ? (
                      <MutedPanel>No live todo items for this session yet.</MutedPanel>
                    ) : (
                      <ul className="space-y-2">
                        {entry.todos.map(todo => (
                          <li
                            className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-3 py-2"
                            key={todo.id}
                          >
                            <div className="flex items-start gap-2">
                              <Badge className={cn('mt-0.5 shrink-0 border text-[10px] uppercase tracking-wide', taskStatusTone(todo.status))}>
                                {taskStatusLabel(todo.status)}
                              </Badge>
                              <div className="min-w-0 flex-1 text-sm text-foreground">{todo.content}</div>
                            </div>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  <div className="space-y-2">
                    <SectionTitle title="Subagents" />
                    {entry.subagents.length === 0 ? (
                      <MutedPanel>No background subagents are running for this session.</MutedPanel>
                    ) : (
                      <div className="space-y-2">
                        {entry.subagents.map(node => (
                          <SubagentTree key={node.id} node={node} />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </section>
            ))}
          </div>
        )}
      </div>
    </PageSearchShell>
  )
}

function SummaryPill({ label, value }: { label: string; value: number }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-3 py-1 text-xs text-(--ui-text-secondary)">
      <span>{label}</span>
      <span className="font-semibold text-foreground">{value}</span>
    </div>
  )
}

function Pill({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn('rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide', className)}>{children}</span>
}

function SectionTitle({ title }: { title: string }) {
  return <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-(--ui-text-tertiary)">{title}</div>
}

function MutedPanel({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-3 py-3 text-sm text-(--ui-text-tertiary)">
      {children}
    </div>
  )
}

function EmptyState({ hasQuery }: { hasQuery: boolean }) {
  return (
    <div className="grid min-h-[18rem] place-items-center rounded-xl border border-dashed border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) px-6 text-center">
      <div className="max-w-md space-y-2">
        <div className="text-base font-semibold text-foreground">{hasQuery ? 'No matching live tasks' : 'No live tasks right now'}</div>
        <p className="text-sm text-(--ui-text-tertiary)">
          {hasQuery
            ? 'Try a different search term, or wait for Hermes to start work in another session.'
            : 'This page lights up automatically when Hermes starts a session, creates todo items, or spins up subagents.'}
        </p>
      </div>
    </div>
  )
}

function SubagentTree({ node, depth = 0 }: { node: SubagentNode; depth?: number }) {
  const latest = latestSubagentLine(node)

  return (
    <div className="space-y-2">
      <div
        className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-3 py-2"
        style={{ marginLeft: depth > 0 ? `${depth * 14}px` : undefined }}
      >
        <div className="flex flex-wrap items-start gap-2">
          <Badge className={cn('border text-[10px] uppercase tracking-wide', subagentStatusTone(node.status))}>
            {subagentStatusLabel(node.status)}
          </Badge>
          <div className="min-w-0 flex-1 space-y-1">
            <div className="text-sm font-medium text-foreground">{node.goal}</div>
            {latest ? <div className="text-xs text-(--ui-text-secondary)">{latest}</div> : null}
            <div className="flex flex-wrap gap-2 text-[11px] text-(--ui-text-tertiary)">
              {node.currentTool ? <span>Tool: {node.currentTool}</span> : null}
              {typeof node.toolCount === 'number' ? <span>Tools used: {node.toolCount}</span> : null}
              {typeof node.durationSeconds === 'number' ? <span>{Math.round(node.durationSeconds)}s</span> : null}
            </div>
          </div>
        </div>
      </div>

      {node.children.map(child => (
        <SubagentTree depth={depth + 1} key={child.id} node={child} />
      ))}
    </div>
  )
}
