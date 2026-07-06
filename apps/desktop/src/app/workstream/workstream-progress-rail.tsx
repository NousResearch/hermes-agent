import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import type { HermesReviewPr } from '@/global'
import type { TodoItem } from '@/lib/todos'
import { cn } from '@/lib/utils'
import { $clarifyRequests, type ClarifyRequest } from '@/store/clarify'
import { $backgroundStatusBySession, type ComposerStatusItem } from '@/store/composer-status'
import { $githubWorkstreamPrBySession } from '@/store/github-workstream'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $subagentsBySession, type SubagentProgress, type SubagentStatus } from '@/store/subagents'
import { $todosBySession } from '@/store/todos'
import { $workflowProgressBySession, type WorkflowPhaseStatus, type WorkflowProgress } from '@/store/workflows'
import { $workstreamActivity, liveWorkstreamSessionId, workstreamCountLabel } from '@/store/workstream'

const EMPTY_SESSION_ID = '__workstream-empty__'

const TODO_STATUS_META = {
  cancelled: { icon: 'circle-slash', tone: 'text-muted-foreground/50' },
  completed: { icon: 'pass-filled', tone: 'text-emerald-500/80' },
  in_progress: { icon: 'loading', tone: 'text-primary/85' },
  pending: { icon: 'circle-outline', tone: 'text-muted-foreground/65' }
} as const

const SUBAGENT_STATUS_META: Record<SubagentStatus, { icon: string; tone: string }> = {
  completed: { icon: 'pass-filled', tone: 'text-emerald-500/80' },
  failed: { icon: 'error', tone: 'text-destructive/85' },
  interrupted: { icon: 'circle-slash', tone: 'text-destructive/75' },
  queued: { icon: 'clock', tone: 'text-muted-foreground/65' },
  running: { icon: 'loading', tone: 'text-primary/85' }
}

const WORKFLOW_STATUS_META: Record<WorkflowPhaseStatus, { icon: string; tone: string }> = {
  completed: { icon: 'pass-filled', tone: 'text-emerald-500/80' },
  failed: { icon: 'error', tone: 'text-destructive/85' },
  pending: { icon: 'circle-outline', tone: 'text-muted-foreground/55' },
  running: { icon: 'loading', tone: 'text-primary/85' },
  skipped: { icon: 'circle-slash', tone: 'text-muted-foreground/45' }
}

function sessionValue<T>(map: Record<string, T>, sessionId: string, liveSessionId: string): T | undefined {
  return liveSessionId === sessionId ? map[sessionId] : (map[liveSessionId] ?? map[sessionId])
}

function StatusIcon({ icon, tone }: { icon: string; tone: string }) {
  if (icon === 'loading') {
    return <GlyphSpinner ariaLabel="running" className={cn('text-[0.78rem]', tone)} spinner="braille" />
  }

  return <Codicon aria-hidden className={tone} name={icon} size="0.78rem" />
}

function Section({ children, count, title }: { children: ReactNode; count?: number; title: string }) {
  return (
    <section className="grid min-w-0 gap-2">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <h3 className="truncate text-[0.66rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground/70">
          {title}
        </h3>
        {typeof count === 'number' ? (
          <span className="shrink-0 rounded-full bg-muted/45 px-1.5 py-0.5 text-[0.6rem] font-medium text-muted-foreground/75">
            {count}
          </span>
        ) : null}
      </div>
      {children}
    </section>
  )
}

function TodoList({ todos }: { todos: TodoItem[] }) {
  return (
    <div className="grid min-w-0 gap-1.5">
      {todos.map(todo => {
        const meta = TODO_STATUS_META[todo.status]

        return (
          <div className="flex min-w-0 items-start gap-2 rounded-md px-1 py-0.5" key={todo.id}>
            <span className="mt-0.5 flex size-3.5 shrink-0 items-center justify-center">
              <StatusIcon icon={meta.icon} tone={meta.tone} />
            </span>
            <span
              className={cn(
                'min-w-0 flex-1 wrap-anywhere text-[0.74rem] leading-relaxed',
                todo.status === 'completed' || todo.status === 'cancelled'
                  ? 'text-muted-foreground/70'
                  : 'text-foreground/90'
              )}
            >
              {todo.content}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function SubagentList({ subagents }: { subagents: SubagentProgress[] }) {
  return (
    <div className="grid min-w-0 gap-3">
      {subagents.map(subagent => {
        const meta = SUBAGENT_STATUS_META[subagent.status]
        const latestStream = subagent.stream.at(-1)

        return (
          <div className="grid min-w-0 gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-muted/15 px-2 py-1.5" key={subagent.id}>
            <div className="flex min-w-0 items-start gap-2">
              <span className="mt-0.5 flex size-3.5 shrink-0 items-center justify-center">
                <StatusIcon icon={meta.icon} tone={meta.tone} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="wrap-anywhere text-[0.74rem] font-medium leading-relaxed text-foreground/90">
                  {subagent.goal}
                </p>
                {subagent.currentTool ? (
                  <p className="truncate font-mono text-[0.63rem] leading-relaxed text-muted-foreground/70">
                    {subagent.currentTool}
                  </p>
                ) : null}
              </div>
            </div>
            {latestStream ? (
              <p className="wrap-anywhere pl-5 text-[0.68rem] leading-relaxed text-muted-foreground/78">
                {latestStream.text}
              </p>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}

function ToolActivityList({ items }: { items: ComposerStatusItem[] }) {
  return (
    <div className="grid min-w-0 gap-1.5">
      {items.map(item => (
        <div className="flex min-w-0 items-center gap-2 rounded-md px-1 py-0.5" key={item.id}>
          <span className={cn('size-1.5 shrink-0 rounded-full', item.state === 'failed' ? 'bg-destructive' : 'bg-primary')} />
          <span className="min-w-0 flex-1 truncate text-[0.72rem] text-foreground/85">{item.title}</span>
        </div>
      ))}
    </div>
  )
}

function NeedsInputCard({ request }: { request?: ClarifyRequest }) {
  return (
    <div className="grid min-w-0 gap-1.5 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2 py-1.5 text-amber-800 dark:text-amber-200">
      <p className="text-[0.74rem] font-medium leading-relaxed">{request?.question || 'Needs your input'}</p>
      {request?.choices?.length ? (
        <div className="flex min-w-0 flex-wrap gap-1">
          {request.choices.map(choice => (
            <span className="max-w-full truncate rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[0.62rem]" key={choice}>
              {choice}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function WorkflowPhaseList({ workflow }: { workflow: WorkflowProgress }) {
  return (
    <div className="grid min-w-0 gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-muted/15 px-2 py-1.5">
      <p className="truncate text-[0.75rem] font-medium text-foreground/90">{workflow.title}</p>
      <div className="grid min-w-0 gap-1.5">
        {workflow.phases.map(phase => {
          const meta = WORKFLOW_STATUS_META[phase.status]

          return (
            <div className="flex min-w-0 items-center gap-2" key={phase.id}>
              <span className="flex size-3.5 shrink-0 items-center justify-center">
                <StatusIcon icon={meta.icon} tone={meta.tone} />
              </span>
              <span className="min-w-0 flex-1 truncate text-[0.72rem] text-foreground/85">{phase.title}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function GithubPrCard({ pr }: { pr: HermesReviewPr }) {
  return (
    <Button
      className="h-auto min-w-0 justify-start gap-2 rounded-lg border-(--ui-stroke-tertiary) bg-muted/15 px-2 py-1.5 text-left"
      onClick={() => void window.hermesDesktop?.openExternal?.(pr.url)}
      type="button"
      variant="outline"
    >
      <Codicon aria-hidden className="shrink-0 text-muted-foreground/80" name="git-pull-request" size="0.78rem" />
      <span className="min-w-0 flex-1 truncate text-[0.74rem] font-medium text-foreground/90">PR #{pr.number}</span>
      <span className="shrink-0 rounded-full bg-muted/45 px-1.5 py-0.5 text-[0.58rem] font-semibold uppercase tracking-[0.08em] text-muted-foreground/75">
        {pr.state}
      </span>
    </Button>
  )
}

interface WorkstreamProgressRailProps {
  onClose?: () => void
  onOpenObservatory?: () => void
}

export function WorkstreamProgressRail({ onClose, onOpenObservatory }: WorkstreamProgressRailProps) {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const sessionId = selectedStoredSessionId || activeSessionId || EMPTY_SESSION_ID
  const liveSessionId = liveWorkstreamSessionId(sessionId, activeSessionId, selectedStoredSessionId)
  const activity = useStore($workstreamActivity(sessionId))
  const todosBySession = useStore($todosBySession)
  const subagentsBySession = useStore($subagentsBySession)
  const backgroundBySession = useStore($backgroundStatusBySession)
  const clarifyRequests = useStore($clarifyRequests)
  const workflowsBySession = useStore($workflowProgressBySession)
  const githubPrBySession = useStore($githubWorkstreamPrBySession)

  const todos = sessionValue(todosBySession, sessionId, liveSessionId) ?? []
  const subagents = sessionValue(subagentsBySession, sessionId, liveSessionId) ?? []
  const backgroundItems = sessionValue(backgroundBySession, sessionId, liveSessionId) ?? []
  const clarifyRequest = sessionValue(clarifyRequests, sessionId, liveSessionId)
  const workflow = sessionValue(workflowsBySession, sessionId, liveSessionId)
  const githubPr = sessionValue(githubPrBySession, sessionId, liveSessionId)

  const hasSelectedSession = sessionId !== EMPTY_SESSION_ID

  const hasLiveWork =
    activity.needsInput ||
    activity.isWorking ||
    todos.length > 0 ||
    subagents.length > 0 ||
    backgroundItems.length > 0 ||
    Boolean(workflow) ||
    Boolean(githubPr)

  return (
    <aside
      aria-label="Workstream progress"
      className="flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) pt-(--titlebar-height) text-(--ui-text-tertiary)"
      data-workstream-progress-rail
      tabIndex={-1}
    >
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-(--ui-stroke-tertiary) px-2.5">
        <span aria-hidden className="text-[0.85rem] leading-none">
          {activity.icon}
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[0.72rem] font-semibold uppercase tracking-[0.12em] text-foreground/80">
            Workstream
          </p>
        </div>
        {onClose ? (
          <Button aria-label="Close workstream panel" onClick={onClose} size="icon-xs" type="button" variant="ghost">
            <Codicon name="close" size="0.78rem" />
          </Button>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain p-3">
        <div className="grid min-w-0 gap-4">
          <div className="grid min-w-0 gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-muted/20 px-2 py-2">
            <div className="flex min-w-0 items-center gap-2">
              <span aria-hidden className="text-[0.9rem] leading-none">
                {activity.icon}
              </span>
              <p className="truncate text-[0.78rem] font-medium text-foreground/90">{activity.label}</p>
            </div>
            <p className="truncate text-[0.66rem] text-muted-foreground/72">
              {hasSelectedSession ? `${workstreamCountLabel(todos.length, 'todo')} · ${workstreamCountLabel(subagents.length, 'agent')}` : 'No session selected'}
            </p>
            {onOpenObservatory ? (
              <Button
                className="mt-1.5 h-auto justify-start gap-1.5 px-1.5 py-1 text-[0.68rem]"
                onClick={onOpenObservatory}
                size="sm"
                type="button"
                variant="ghost"
              >
                <Codicon aria-hidden name="graph-line" size="0.72rem" />
                Open Observatory
              </Button>
            ) : null}
          </div>

          {!hasLiveWork ? (
            <div className="grid place-items-center rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-8 text-center">
              <p className="text-[0.75rem] font-medium text-muted-foreground/80">
                {hasSelectedSession ? 'No live work for this session' : 'Select a session to see live work'}
              </p>
            </div>
          ) : (
            <>
              {activity.needsInput ? (
                <Section title="Needs input">
                  <NeedsInputCard request={clarifyRequest} />
                </Section>
              ) : null}

              {todos.length > 0 ? (
                <Section count={todos.length} title="Todos">
                  <TodoList todos={todos} />
                </Section>
              ) : null}

              {subagents.length > 0 ? (
                <Section count={subagents.length} title="Subagents">
                  <SubagentList subagents={subagents} />
                </Section>
              ) : null}

              {backgroundItems.length > 0 ? (
                <Section count={backgroundItems.length} title="Tool activity">
                  <ToolActivityList items={backgroundItems} />
                </Section>
              ) : null}

              {workflow ? (
                <Section count={workflow.phases.length} title="Workflow phases">
                  <WorkflowPhaseList workflow={workflow} />
                </Section>
              ) : null}

              {githubPr ? (
                <Section title="GitHub">
                  <GithubPrCard pr={githubPr} />
                </Section>
              ) : null}
            </>
          )}
        </div>
      </div>
    </aside>
  )
}
