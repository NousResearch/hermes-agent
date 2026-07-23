import { useQuery } from '@tanstack/react-query'
import { Fragment, type ReactNode, useState } from 'react'
import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { SidebarPanelLabel } from '../../shell/sidebar-label'
import { RightSidebarSectionHeader } from '../index'

interface KanbanBoard {
  archived?: boolean
  counts?: Record<string, number>
  description?: string
  is_current?: boolean
  name?: string
  slug: string
  total?: number
}

interface KanbanTask {
  assignee?: null | string
  body?: null | string
  branch_name?: null | string
  created_at?: null | number
  current_step_key?: null | string
  id: string
  priority?: null | number
  skills?: string[]
  status?: null | string
  tenant?: null | string
  title: string
  workspace_kind?: null | string
  workspace_path?: null | string
  workflow_template_id?: null | string
}

interface CliExecResponse {
  blocked?: boolean
  code: number
  hint?: string
  output: string
}

interface KanbanTabProps {
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

interface KanbanSnapshot {
  boards: KanbanBoard[]
  currentBoard: KanbanBoard | null
  tasks: KanbanTask[]
}

const KANBAN_QUERY_KEY = ['right-sidebar', 'kanban'] as const

function parseJsonOutput<T>(label: string, result: CliExecResponse): T {
  if (result.blocked) {
    throw new Error(result.hint || `${label} is not available in desktop mode`)
  }

  if (result.code !== 0) {
    throw new Error(result.output || `${label} failed`)
  }

  try {
    return JSON.parse(result.output) as T
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`${label}: ${message}`)
  }
}

async function loadKanbanSnapshot(requestGateway: KanbanTabProps['requestGateway']): Promise<KanbanSnapshot> {
  const [boardsResult, tasksResult] = await Promise.all([
    requestGateway<CliExecResponse>('cli.exec', { argv: ['kanban', 'boards', 'list', '--json'] }),
    requestGateway<CliExecResponse>('cli.exec', { argv: ['kanban', 'list', '--json'] })
  ])

  const boards = parseJsonOutput<KanbanBoard[]>('kanban boards', boardsResult)
  const tasks = parseJsonOutput<KanbanTask[]>('kanban list', tasksResult)
  const currentBoard = boards.find(board => board.is_current) ?? boards[0] ?? null

  return { boards, currentBoard, tasks }
}

function statusTone(status: string | null | undefined): 'default' | 'destructive' | 'muted' | 'outline' | 'warn' {
  switch ((status || '').toLowerCase()) {
    case 'blocked':
      return 'destructive'

    case 'running':

    case 'in_progress':
      return 'warn'

    case 'done':

    case 'completed':
      return 'default'

    case 'ready':
      return 'outline'

    default:
      return 'muted'
  }
}

function formatTimestamp(epoch: null | number | undefined): string | null {
  if (!epoch) {
    return null
  }

  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short'
    }).format(new Date(epoch * 1000))
  } catch {
    return null
  }
}

function statusSummary(tasks: KanbanTask[]): Map<string, number> {
  const counts = new Map<string, number>()

  for (const task of tasks) {
    const key = (task.status || 'unknown').toLowerCase()
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }

  return counts
}

export function KanbanTab({ requestGateway }: KanbanTabProps) {
  const { t } = useI18n()
  const r = t.rightSidebar.kanbanPanel
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)

  const query = useQuery({
    queryFn: () => loadKanbanSnapshot(requestGateway),
    queryKey: KANBAN_QUERY_KEY,
    refetchInterval: 30_000
  })

  const snapshot = query.data
  const tasks = snapshot?.tasks ?? []
  const selectedTask = tasks.find(task => task.id === selectedTaskId) ?? tasks[0] ?? null
  const counts = statusSummary(tasks)

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <RightSidebarSectionHeader>
        <SidebarPanelLabel>{r.title}</SidebarPanelLabel>
        <div className="ml-auto flex items-center gap-1">
          {snapshot?.currentBoard ? (
            <Badge className="max-w-32 truncate" title={snapshot.currentBoard.slug} variant="muted">
              {snapshot.currentBoard.name || snapshot.currentBoard.slug}
            </Badge>
          ) : null}
          <Tip label={r.refresh}>
            <Button
              aria-label={r.refresh}
              className="text-sidebar-foreground/70 hover:bg-sidebar-accent! hover:text-sidebar-accent-foreground! focus-visible:ring-sidebar-ring"
              onClick={() => void query.refetch()}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="refresh" size="0.8125rem" spinning={query.isFetching} />
            </Button>
          </Tip>
        </div>
      </RightSidebarSectionHeader>

      {query.isLoading ? (
        <PageLoader className="min-h-24 px-3" label={r.loading} />
      ) : query.error ? (
        <KanbanEmptyState body={query.error.message} title={r.loadFailed} />
      ) : !snapshot?.currentBoard ? (
        <KanbanEmptyState body={<InlineCodeText text={r.noBoardsBody} />} title={r.noBoardsTitle} />
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="px-3 pb-2">
            <div className="text-[0.72rem] font-medium text-(--ui-text-secondary)">
              {snapshot.currentBoard.name || snapshot.currentBoard.slug}
            </div>
            {snapshot.currentBoard.description ? (
              <p className="mt-1 text-[0.67rem] leading-relaxed text-(--ui-text-tertiary)">
                {snapshot.currentBoard.description}
              </p>
            ) : null}
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Array.from(counts.entries()).map(([status, total]) => (
                <Badge key={status} variant={statusTone(status)}>
                  {r.statusCount(status, total)}
                </Badge>
              ))}
              {counts.size === 0 ? <Badge variant="outline">{r.emptyBadge}</Badge> : null}
            </div>
          </div>

          <Separator />

          {tasks.length === 0 ? (
            <KanbanEmptyState body={r.noTasksBody} title={r.noTasksTitle} />
          ) : (
            <div className="flex min-h-0 flex-1 flex-col">
              <ScrollArea className="min-h-0 flex-1">
                <div className="space-y-1 p-2">
                  {tasks.map(task => {
                    const active = selectedTask?.id === task.id

                    return (
                      <button
                        className={cn(
                          'w-full rounded-md border border-transparent px-2.5 py-2 text-left transition hover:border-(--ui-stroke-secondary) hover:bg-(--ui-control-hover-background)',
                          active && 'border-(--ui-stroke-secondary) bg-(--ui-control-active-background)'
                        )}
                        key={task.id}
                        onClick={() => setSelectedTaskId(task.id)}
                        type="button"
                      >
                        <div className="flex items-start gap-2">
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[0.72rem] font-medium text-(--ui-text-secondary)">
                              {task.title}
                            </div>
                            <div className="mt-1 flex flex-wrap items-center gap-1.5">
                              <Badge variant={statusTone(task.status)}>{task.status || r.unknownStatus}</Badge>
                              <Badge variant="outline">{task.id}</Badge>
                              {task.assignee ? <Badge variant="muted">{task.assignee}</Badge> : null}
                            </div>
                          </div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </ScrollArea>

              <Separator />

              {/* TODO: Keep the MVP read-only until desktop mutation affordances land; task actions still flow through chat/CLI. */}
              {selectedTask ? <KanbanTaskDetails task={selectedTask} /> : null}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function KanbanTaskDetails({ task }: { task: KanbanTask }) {
  const { t } = useI18n()
  const r = t.rightSidebar.kanbanPanel
  const createdAt = formatTimestamp(task.created_at)

  return (
    <div className="shrink-0 space-y-2 px-3 py-2.5">
      <div className="text-[0.68rem] font-semibold uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
        {r.details}
      </div>
      <div className="text-[0.75rem] font-medium text-(--ui-text-secondary)">{task.title}</div>
      <div className="flex flex-wrap gap-1.5">
        <Badge variant={statusTone(task.status)}>{task.status || r.unknownStatus}</Badge>
        <Badge variant="outline">{task.id}</Badge>
        {task.assignee ? <Badge variant="muted">{task.assignee}</Badge> : null}
      </div>
      {task.body ? <p className="text-[0.69rem] leading-relaxed text-(--ui-text-tertiary)">{task.body}</p> : null}
      <div className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-[0.67rem] text-(--ui-text-tertiary)">
        <span>{r.workspace}</span>
        <span className="min-w-0 truncate">
          {task.workspace_path
            ? `${task.workspace_kind || 'workspace'} · ${task.workspace_path}`
            : task.workspace_kind || r.notSet}
        </span>
        <span>{r.created}</span>
        <span>{createdAt || r.notSet}</span>
        <span>{r.branch}</span>
        <span className="truncate">{task.branch_name || r.notSet}</span>
        <span>{r.workflow}</span>
        <span className="truncate">{task.workflow_template_id || task.current_step_key || r.notSet}</span>
      </div>
      {task.skills?.length ? (
        <div className="flex flex-wrap gap-1.5">
          {task.skills.map(skill => (
            <Badge key={skill} variant="outline">
              {skill}
            </Badge>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function InlineCodeText({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`)/g).filter(Boolean)

  return (
    <>
      {parts.map((part, index) =>
        part.startsWith('`') && part.endsWith('`') ? (
          <code
            className="mx-px rounded bg-muted/50 px-1 py-px font-mono text-[0.92em] text-muted-foreground/85"
            key={`code-${index}`}
          >
            {part.slice(1, -1)}
          </code>
        ) : (
          <Fragment key={`text-${index}`}>{part}</Fragment>
        )
      )}
    </>
  )
}

function KanbanEmptyState({ body, title }: { body: ReactNode; title: string }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-1 px-4 text-center">
      <div className="text-[0.7rem] font-semibold uppercase tracking-[0.07em] text-muted-foreground/75">{title}</div>
      <div className="text-[0.68rem] leading-relaxed text-muted-foreground/65">{body}</div>
    </div>
  )
}

export { loadKanbanSnapshot }
