import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { BrailleSpinner } from '@/components/ui/braille-spinner'
import { type Translations, useI18n } from '@/i18n'
import { AlertCircle, CheckCircle2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $tasksBySession,
  type AgentTask,
  isTerminalTaskStatus,
  pruneExpiredAgentTasks,
  visibleAgentTasks
} from '@/store/tasks'

// Same vocabulary as statusGlyph() in ./index.tsx, mapped onto the Action
// Runtime's TaskStatus values (succeeded/partial read as done; needs_input
// and blocked read as attention, like failed, since both ended without one).
function taskGlyph(status: AgentTask['status'], a: Translations['agents']): ReactNode {
  if (status === 'running') {
    return (
      <BrailleSpinner
        ariaLabel={a.running}
        className="size-3.5 shrink-0 text-[0.95rem] text-muted-foreground/80"
        spinner="breathe"
      />
    )
  }

  if (status === 'succeeded' || status === 'partial') {
    return (
      <CheckCircle2 aria-label={a.done} className="size-3.5 shrink-0 text-emerald-600/85 dark:text-emerald-400/85" />
    )
  }

  return <AlertCircle aria-label={a.failed} className="size-3.5 shrink-0 text-destructive" />
}

// Mirrors fmtAge() in ./index.tsx (kept local: index.tsx imports this file).
const fmtAge = (at: number, nowMs: number, a: Translations['agents']) => {
  const s = Math.max(0, Math.round((nowMs - at) / 1000))

  if (s < 2) {
    return a.ageNow
  }

  if (s < 60) {
    return a.ageSeconds(s)
  }

  const m = Math.floor(s / 60)

  if (m < 60) {
    return a.ageMinutes(m)
  }

  return a.ageHours(Math.floor(m / 60))
}

function TaskRow({ nowMs, task }: { nowMs: number; task: AgentTask }) {
  const { t } = useI18n()
  const terminal = isTerminalTaskStatus(task.status)

  const meta = [
    terminal ? task.status : '',
    task.toolCount > 0 ? t.agents.toolsCount(task.toolCount) : '',
    task.lastTool ?? '',
    fmtAge(terminal ? (task.finishedAt ?? task.startedAt) : task.startedAt, nowMs, t.agents)
  ].filter(Boolean)

  return (
    <div
      className="flex min-w-0 items-center gap-2"
      data-slot="task-row"
      title={[task.goal, task.error].filter(Boolean).join(' — ') || undefined}
    >
      <span className="flex h-[1.1rem] shrink-0 items-center">{taskGlyph(task.status, t.agents)}</span>
      {task.intent ? <Badge variant="outline">{task.intent}</Badge> : null}
      <span
        className={cn(
          'min-w-0 flex-1 truncate text-[0.78rem] leading-[1.1rem] text-foreground/90',
          task.status === 'running' && 'shimmer text-foreground/65'
        )}
      >
        {task.label || task.goal}
      </span>
      {meta.length > 0 ? (
        <span className="shrink-0 font-mono text-[0.64rem] leading-[1.05rem] text-muted-foreground/65">
          {meta.join(' · ')}
        </span>
      ) : null}
    </div>
  )
}

/**
 * Compact board over the gateway's live task registry: one row per active
 * task, fed by the task.list seed + task.started/task.completed push events
 * ($tasksBySession). Terminal rows linger TASK_LINGER_MS, then drop. Renders
 * nothing while the registry is quiet so the spawn tree keeps its empty state.
 */
interface TaskBoardProps {
  sessionId: null | string
}

export function TaskBoard({ sessionId }: TaskBoardProps) {
  const { t } = useI18n()
  const tasksBySession = useStore($tasksBySession)
  const sessionTasks = sessionId ? (tasksBySession[sessionId] ?? []) : []
  const hasTasks = sessionTasks.length > 0
  const [nowMs, setNowMs] = useState(() => Date.now())
  const rows = useMemo(() => visibleAgentTasks(tasksBySession, nowMs, sessionId), [nowMs, sessionId, tasksBySession])

  // The tick both refreshes ages and drops terminal rows once they expire —
  // it must keep running while only terminal rows remain.
  useEffect(() => {
    if (!hasTasks || typeof window === 'undefined') {
      return
    }

    const id = window.setInterval(() => {
      const nextNow = Date.now()

      pruneExpiredAgentTasks(nextNow)
      setNowMs(nextNow)
    }, 1_000)

    return () => window.clearInterval(id)
  }, [hasTasks])

  if (rows.length === 0) {
    return null
  }

  return (
    <section className="mb-4 grid shrink-0 gap-2" data-slot="task-board">
      <p className="text-[0.66rem] font-medium uppercase tracking-wider text-muted-foreground/70">
        {t.agents.tasksTitle}
      </p>
      <div className="grid min-w-0 gap-1.5">
        {rows.map(task => (
          <TaskRow key={`${task.sessionId}:${task.id}`} nowMs={nowMs} task={task} />
        ))}
      </div>
    </section>
  )
}
