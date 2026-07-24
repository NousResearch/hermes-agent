import { useStore } from '@nanostores/react'

import { revealTreePane } from '@/components/pane-shell/tree/store'
import { Codicon } from '@/components/ui/codicon'
import { type Translations, useI18n } from '@/i18n'
import type { TodoItem } from '@/lib/todos'
import { cn } from '@/lib/utils'
import { $activeSessionId } from '@/store/session'
import { $todosBySession } from '@/store/todos'

export type RunBoardKind = 'active' | 'blockedNeedsYou' | 'blockedTechnical' | 'done' | 'empty' | 'ready' | 'waiting'

export interface RunBoardState {
  completed: number
  current: TodoItem | null
  kind: RunBoardKind
  supportNeeded: boolean
  total: number
}

const NEEDS_YOU = /\bblocked\s*:\s*needs\s+you\b/i
const TECHNICAL_BLOCK = /\bblocked\s*:\s*technical\b/i
const WAITING = /^\s*waiting(?:\s*:|\b)/i

export function deriveRunBoardState(todos: readonly TodoItem[]): RunBoardState {
  const completed = todos.filter(todo => todo.status === 'completed' || todo.status === 'cancelled').length

  const current =
    todos.find(todo => todo.status === 'in_progress') ?? todos.find(todo => todo.status === 'pending') ?? null

  if (todos.length === 0) {
    return { completed: 0, current: null, kind: 'empty', supportNeeded: false, total: 0 }
  }

  if (!current) {
    return { completed, current: null, kind: 'done', supportNeeded: false, total: todos.length }
  }

  const content = current.content.trim()

  if (NEEDS_YOU.test(content)) {
    return { completed, current, kind: 'blockedNeedsYou', supportNeeded: true, total: todos.length }
  }

  if (TECHNICAL_BLOCK.test(content)) {
    return { completed, current, kind: 'blockedTechnical', supportNeeded: false, total: todos.length }
  }

  if (WAITING.test(content)) {
    return { completed, current, kind: 'waiting', supportNeeded: false, total: todos.length }
  }

  return {
    completed,
    current,
    kind: current.status === 'pending' ? 'ready' : 'active',
    supportNeeded: false,
    total: todos.length
  }
}

const STATE_TONE: Record<RunBoardKind, string> = {
  active: 'border-sky-500/35 bg-sky-500/10 text-sky-300',
  blockedNeedsYou: 'border-amber-500/40 bg-amber-500/12 text-amber-300',
  blockedTechnical: 'border-red-500/35 bg-red-500/10 text-red-300',
  done: 'border-emerald-500/35 bg-emerald-500/10 text-emerald-300',
  empty: 'border-(--ui-stroke-secondary) bg-(--ui-bg-subtle) text-(--ui-text-muted)',
  ready: 'border-violet-500/35 bg-violet-500/10 text-violet-300',
  waiting: 'border-cyan-500/35 bg-cyan-500/10 text-cyan-300'
}

function stateLabel(kind: RunBoardKind, copy: Translations['runBoard']): string {
  return {
    active: copy.active,
    blockedNeedsYou: copy.blockedNeedsYou,
    blockedTechnical: copy.blockedTechnical,
    done: copy.done,
    empty: copy.ready,
    ready: copy.ready,
    waiting: copy.waiting
  }[kind]
}

function TaskGlyph({ status }: { status: TodoItem['status'] }) {
  if (status === 'in_progress') {
    return <Codicon className="shrink-0 text-sky-400" name="loading" size="0.9rem" spinning />
  }

  if (status === 'completed') {
    return <Codicon className="shrink-0 text-emerald-400" name="pass-filled" size="0.9rem" />
  }

  if (status === 'cancelled') {
    return <Codicon className="shrink-0 text-(--ui-text-muted)" name="circle-slash" size="0.9rem" />
  }

  return <span aria-hidden className="size-2.5 shrink-0 rounded-full border border-dashed border-(--ui-text-muted)" />
}

export function RunBoardPane() {
  const { t } = useI18n()
  const activeSessionId = useStore($activeSessionId)
  const todosBySession = useStore($todosBySession)
  const todos = activeSessionId ? (todosBySession[activeSessionId] ?? []) : []
  const state = deriveRunBoardState(todos)
  const copy = t.runBoard

  return (
    <section className="flex h-full min-h-0 flex-col bg-(--ui-bg-base)" data-testid="run-board">
      <header className="shrink-0 border-b border-(--ui-stroke-secondary) px-3 py-3">
        <div className="mb-2 flex items-center gap-1.5 text-(--ui-text-secondary)">
          <Codicon className="text-(--ui-text-muted)" name="checklist" size="0.82rem" />
          <h2 className="text-[0.68rem] font-semibold uppercase tracking-[0.12em]">{copy.title}</h2>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span
            className={cn(
              'min-w-0 rounded-full border px-2 py-0.5 text-[0.65rem] font-semibold tracking-[0.08em]',
              STATE_TONE[state.kind]
            )}
          >
            {stateLabel(state.kind, copy)}
          </span>
          {state.total > 0 && (
            <span className="shrink-0 text-[0.66rem] tabular-nums text-(--ui-text-muted)">
              {copy.progress(state.completed, state.total)}
            </span>
          )}
        </div>
        <p
          className={cn(
            'mt-2 text-[0.7rem] font-medium',
            state.supportNeeded ? 'text-amber-300' : 'text-(--ui-text-secondary)'
          )}
        >
          {copy.supportNeeded(state.supportNeeded ? copy.supportYes : copy.supportNo)}
        </p>
      </header>

      {todos.length === 0 ? (
        <div className="flex min-h-0 flex-1 items-center justify-center px-5 text-center">
          <p className="text-xs leading-5 text-(--ui-text-muted)">{copy.empty}</p>
        </div>
      ) : (
        <ol className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
          {todos.map(todo => {
            const isCurrent = state.current?.id === todo.id

            return (
              <li
                aria-current={isCurrent ? 'step' : undefined}
                className={cn(
                  'mb-1 flex items-start gap-2 rounded-lg border px-2.5 py-2 text-xs leading-4',
                  isCurrent
                    ? 'border-sky-500/30 bg-sky-500/8 text-(--ui-text-primary)'
                    : 'border-transparent text-(--ui-text-secondary)',
                  (todo.status === 'completed' || todo.status === 'cancelled') && 'opacity-65'
                )}
                key={todo.id}
              >
                <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center">
                  <TaskGlyph status={todo.status} />
                </span>
                <span className={cn('min-w-0 break-words', todo.status === 'completed' && 'line-through decoration-1')}>
                  {todo.content}
                </span>
              </li>
            )
          })}
        </ol>
      )}
    </section>
  )
}

/** Compact companion that remains in fixed chrome when the right side is
 * deliberately collapsed. Clicking it restores and fronts the Run Board. */
export function RunBoardStatusbarItem() {
  const { t } = useI18n()
  const activeSessionId = useStore($activeSessionId)
  const todosBySession = useStore($todosBySession)
  const todos = activeSessionId ? (todosBySession[activeSessionId] ?? []) : []
  const state = deriveRunBoardState(todos)
  const copy = t.runBoard

  return (
    <button
      className="inline-flex h-full items-center gap-1.5 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
      onClick={() => revealTreePane('run-board')}
      title={copy.title}
      type="button"
    >
      <Codicon
        className={state.supportNeeded ? 'text-amber-300' : 'text-(--ui-text-muted)'}
        name="checklist"
        size="0.78rem"
      />
      <span className="truncate">{copy.title}</span>
      <span className={cn('truncate font-medium', state.supportNeeded && 'text-amber-300')}>
        {stateLabel(state.kind, copy)}
      </span>
    </button>
  )
}
