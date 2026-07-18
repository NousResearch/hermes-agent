import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { useId, useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import {
  createTodoHistoryMessagesSelector,
  sessionTodoHistory,
  type TodoHistoryMessage,
  type TodoHistorySnapshot,
  type TodoItem
} from '@/lib/todos'
import { cn } from '@/lib/utils'
import { $todosBySession } from '@/store/todos'

interface TaskHistoryPopoverProps {
  busy: boolean
  liveTodos: readonly TodoItem[] | null
  messages: readonly TodoHistoryMessage[]
  sessionId: string | null
}

const INITIAL_HISTORY_LIMIT = 10

const doneCount = (todos: readonly TodoItem[]) =>
  todos.filter(todo => todo.status === 'completed' || todo.status === 'cancelled').length

function TodoGlyph({ snapshot, todo }: { snapshot: TodoHistorySnapshot; todo: TodoItem }) {
  const completed = todo.status === 'completed'
  const cancelled = todo.status === 'cancelled'

  if (completed || cancelled) {
    return (
      <Codicon
        aria-hidden
        className={completed ? 'text-emerald-500/80' : 'text-muted-foreground/45'}
        name={completed ? 'pass-filled' : 'circle-slash'}
        size="0.8rem"
      />
    )
  }

  return (
    <span
      aria-hidden
      className={cn(
        'box-border size-[0.7rem] rounded-full border border-dashed',
        snapshot.state === 'running' ? 'border-primary/70' : 'border-muted-foreground/55'
      )}
    />
  )
}

function SnapshotSection({ label, snapshot }: { label: string; snapshot: TodoHistorySnapshot }) {
  const copy = useI18n().t.statusStack.taskHistory
  const timestamp = snapshot.timestamp ? new Date(snapshot.timestamp * 1000) : null

  const stateLabel =
    snapshot.state === 'running' ? copy.running : snapshot.state === 'unfinished' ? copy.unfinished : copy.completed

  return (
    <section aria-label={label} className="grid gap-1.5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-xs font-medium text-(--ui-text-primary)">{label}</h3>
          {timestamp && (
            <time
              className="block text-[0.65rem] leading-4 text-(--ui-text-tertiary)"
              dateTime={timestamp.toISOString()}
            >
              {timestamp.toLocaleString()}
            </time>
          )}
        </div>
        <span
          className={cn(
            'text-[0.65rem] leading-4',
            snapshot.state === 'running' ? 'text-primary' : 'text-(--ui-text-tertiary)'
          )}
        >
          {stateLabel}
        </span>
      </div>
      <ul className="grid gap-1">
        {snapshot.todos.map(todo => (
          <li className="flex min-w-0 items-start gap-2 text-[0.72rem] leading-4" key={todo.id}>
            <span className="mt-[0.15rem] grid size-3.5 shrink-0 place-items-center">
              <TodoGlyph snapshot={snapshot} todo={todo} />
            </span>
            <span
              className={cn(
                'min-w-0 wrap-break-word',
                todo.status === 'completed' || todo.status === 'cancelled'
                  ? 'text-(--ui-text-tertiary)'
                  : 'text-(--ui-text-primary)'
              )}
            >
              {todo.content}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}

/** Transcript-derived history plus the existing live todo store as a temporary overlay. */
export function TaskHistoryPopover({ busy, liveTodos, messages, sessionId }: TaskHistoryPopoverProps) {
  const { t } = useI18n()
  const copy = t.statusStack.taskHistory
  const generatedId = useId()
  const panelId = `task-history-${generatedId.replaceAll(':', '')}`
  const [openBySession, setOpenBySession] = useState<Record<string, boolean>>({})
  const [visibleBySession, setVisibleBySession] = useState<Record<string, number>>({})
  const history = useMemo(() => sessionTodoHistory(messages, liveTodos, busy), [busy, liveTodos, messages])
  const latest = history.current ?? history.snapshots[0]

  if (!sessionId || !latest) {
    return null
  }

  const open = openBySession[sessionId] ?? false
  const setOpen = (next: boolean) => setOpenBySession(current => ({ ...current, [sessionId]: next }))
  const snapshots = history.current ? [history.current, ...history.snapshots] : history.snapshots
  const visibleCount = visibleBySession[sessionId] ?? INITIAL_HISTORY_LIMIT
  const visibleSnapshots = snapshots.slice(0, visibleCount)
  const label = t.statusStack.todos(doneCount(latest.todos), latest.todos.length)

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <Button
          aria-controls={panelId}
          aria-expanded={open}
          aria-label={label}
          size="xs"
          type="button"
          variant="outline"
        >
          {label}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" aria-label={copy.title} id={panelId} role="dialog" side="top">
        <div className="mb-2 flex items-center justify-between gap-3">
          <h2 className="text-xs font-semibold text-(--ui-text-primary)">{copy.title}</h2>
          <Tip label={copy.close}>
            <Button aria-label={copy.close} onClick={() => setOpen(false)} size="icon-xs" type="button" variant="ghost">
              <Codicon name="close" size="0.75rem" />
            </Button>
          </Tip>
        </div>
        <div className="grid max-h-80 gap-3 overflow-y-auto pr-1">
          {visibleSnapshots.map((snapshot, index) => (
            <SnapshotSection
              key={`${snapshot.state}:${snapshot.id}`}
              label={snapshot.state === 'running' ? copy.current : index === 0 ? copy.latest : copy.previous}
              snapshot={snapshot}
            />
          ))}
          {visibleSnapshots.length < snapshots.length && (
            <Button
              className="justify-self-start"
              onClick={() =>
                setVisibleBySession(current => ({ ...current, [sessionId]: visibleCount + INITIAL_HISTORY_LIMIT }))
              }
              size="xs"
              type="button"
              variant="ghost"
            >
              {copy.loadOlder}
            </Button>
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}

export function SessionTaskHistoryPopover({ busy, sessionId }: { busy: boolean; sessionId: string | null }) {
  const view = useSessionView()

  const todoMessagesStore = useMemo(
    () => computed(view.$messages, createTodoHistoryMessagesSelector()),
    [view.$messages]
  )

  const messages = useStore(todoMessagesStore)
  const todosBySession = useStore($todosBySession)

  return (
    <TaskHistoryPopover
      busy={busy}
      liveTodos={sessionId ? (todosBySession[sessionId] ?? null) : null}
      messages={messages}
      sessionId={sessionId}
    />
  )
}
