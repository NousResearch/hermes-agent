import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $desktopActionTasks, buildRailTasks, type RailTask, type RailTaskStatus } from '@/store/activity'
import { $previewServerRestart } from '@/store/preview'
import { $sessions, $unreadFinishedSessionIds } from '@/store/session'
import { $workingSessionIds } from '@/store/session-states'

interface BackgroundWorkPanelProps {
  onOpenSession: (sessionId: string) => void
  tasks: RailTask[]
}

export function BackgroundWorkRail({ onOpenSession }: { onOpenSession: (sessionId: string) => void }) {
  const workingSessionIds = useStore($workingSessionIds)
  const finishedSessionIds = useStore($unreadFinishedSessionIds)
  const sessions = useStore($sessions)
  const previewRestart = useStore($previewServerRestart)
  const actionTasks = useStore($desktopActionTasks)

  const tasks = useMemo(
    () => buildRailTasks(workingSessionIds, finishedSessionIds, sessions, previewRestart, actionTasks),
    [actionTasks, finishedSessionIds, previewRestart, sessions, workingSessionIds]
  )

  return <BackgroundWorkPanel onOpenSession={onOpenSession} tasks={tasks} />
}

export function BackgroundWorkPanel({ onOpenSession, tasks }: BackgroundWorkPanelProps) {
  const { t } = useI18n()
  const copy = t.rightSidebar
  const [expanded, setExpanded] = useState(true)
  const running = tasks.filter(task => task.status === 'running')
  const finished = tasks.filter(task => task.status !== 'running')

  return (
    <section
      aria-label={copy.backgroundWork}
      className="flex max-h-[45%] min-h-11 shrink-0 flex-col border-b border-(--ui-stroke-tertiary)"
    >
      <button
        aria-controls="background-work-list"
        aria-expanded={expanded}
        aria-label={expanded ? copy.collapseBackgroundWork : copy.expandBackgroundWork}
        className="flex h-11 shrink-0 items-center gap-2 px-2.5 text-left text-(--ui-text-secondary) transition hover:bg-(--ui-control-hover-background) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-sidebar-ring"
        onClick={() => setExpanded(value => !value)}
        type="button"
      >
        <Codicon className="text-(--ui-text-tertiary)" name="server-process" size="0.8125rem" />
        <span className="min-w-0 flex-1 truncate text-xs font-medium">{copy.backgroundWork}</span>
        <span className="rounded-full bg-(--ui-control-active-background) px-1.5 py-0.5 font-mono text-[0.625rem] tabular-nums text-(--ui-text-tertiary)">
          {tasks.length}
        </span>
        <Codicon
          className="text-(--ui-text-quaternary)"
          name={expanded ? 'chevron-up' : 'chevron-down'}
          size="0.75rem"
        />
      </button>

      {expanded && (
        <div className="min-h-0 overflow-x-hidden overflow-y-auto overscroll-contain px-2 pb-2" id="background-work-list">
          {tasks.length === 0 ? (
            <p className="grid min-h-11 place-items-center text-xs text-(--ui-text-quaternary)">
              {copy.noBackgroundWork}
            </p>
          ) : (
            <div className="grid gap-2">
              {running.length > 0 && (
                <WorkGroup label={copy.runningWork} onOpenSession={onOpenSession} tasks={running} />
              )}
              {finished.length > 0 && (
                <WorkGroup label={copy.finishedWork} onOpenSession={onOpenSession} tasks={finished} />
              )}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

function WorkGroup({
  label,
  onOpenSession,
  tasks
}: {
  label: string
  onOpenSession: (sessionId: string) => void
  tasks: RailTask[]
}) {
  return (
    <section>
      <h3 className="px-1 py-1 text-[0.625rem] font-medium uppercase tracking-wider text-(--ui-text-quaternary)">
        {label}
      </h3>
      <div className="grid gap-0.5">
        {tasks.map(task => (
          <WorkRow key={task.id} onOpenSession={onOpenSession} task={task} />
        ))}
      </div>
    </section>
  )
}

function WorkRow({ onOpenSession, task }: { onOpenSession: (sessionId: string) => void; task: RailTask }) {
  const { t } = useI18n()
  const copy = t.rightSidebar

  const content = (
    <>
      <StatusGlyph status={task.status} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs text-(--ui-text-secondary)">{task.label}</span>
        <span className="block truncate text-[0.6875rem] text-(--ui-text-quaternary)">
          {statusLabel(task.status, copy)}
          {task.detail ? ` · ${task.detail}` : ''}
        </span>
      </span>
      {task.sessionId && <Codicon className="text-(--ui-text-quaternary)" name="chevron-right" size="0.75rem" />}
    </>
  )

  const classes = cn(
    'flex min-h-11 w-full items-center gap-2 rounded-md px-2 text-left',
    task.sessionId &&
      'transition hover:bg-(--ui-control-hover-background) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring'
  )

  return task.sessionId ? (
    <button
      aria-label={copy.openBackgroundSession(task.label)}
      className={classes}
      onClick={() => onOpenSession(task.sessionId!)}
      type="button"
    >
      {content}
    </button>
  ) : (
    <div className={classes}>{content}</div>
  )
}

function StatusGlyph({ status }: { status: RailTaskStatus }) {
  if (status === 'running') {
    return <Codicon className="shrink-0 text-(--ui-accent)" name="loading" size="0.75rem" spinning />
  }

  return (
    <Codicon
      className={cn('shrink-0', status === 'error' ? 'text-destructive' : 'text-emerald-500')}
      name={status === 'error' ? 'error' : 'check'}
      size="0.75rem"
    />
  )
}

function statusLabel(status: RailTaskStatus, copy: ReturnType<typeof useI18n>['t']['rightSidebar']): string {
  if (status === 'running') {
    return copy.workRunning
  }

  return status === 'error' ? copy.workFailed : copy.workCompleted
}
