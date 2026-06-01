import { type FC } from 'react'

import { Checkbox } from '@/components/ui/checkbox'
import { Loader2Icon } from '@/lib/icons'
import { parseTodos, type TodoItem, type TodoStatus } from '@/lib/todos'
import { cn } from '@/lib/utils'

export function todosFromMessageContent(content: unknown): TodoItem[] {
  if (!Array.isArray(content)) {
    return []
  }

  let latest: null | TodoItem[] = null

  for (const part of content) {
    if (!part || typeof part !== 'object') {
      continue
    }

    const row = part as Record<string, unknown>

    if (row.type !== 'tool-call' || row.toolName !== 'todo') {
      continue
    }

    const parsed = parseTodos(row.result) ?? parseTodos(row.args)

    if (parsed !== null) {
      latest = parsed
    }
  }

  return latest ?? []
}

const headerLabel = (todos: readonly TodoItem[]): string =>
  todos.find(t => t.status === 'in_progress')?.content ??
  todos.find(t => t.status === 'pending')?.content ??
  todos.at(-1)?.content ??
  'Tasks'

const Checkmark: FC<{ status: TodoStatus; label: string }> = ({ status, label }) => {
  if (status === 'in_progress') {
    return (
      <span
        aria-label={`In progress: ${label}`}
        className="grid size-[1.1rem] shrink-0 place-items-center rounded-full border border-ring/65 bg-[color-mix(in_srgb,var(--dt-ring)_14%,transparent)]"
      >
        <Loader2Icon className="size-3 animate-spin text-ring" />
      </span>
    )
  }

  const checked = status === 'completed'

  return (
    <Checkbox
      aria-label={label}
      checked={checked}
      className={cn(
        'size-[1.1rem] shrink-0 rounded-full border-border/80 pointer-events-none disabled:cursor-default disabled:opacity-100',
        checked &&
          'data-[state=checked]:border-primary data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground [&_[data-slot=checkbox-indicator]_svg]:size-3',
        status === 'cancelled' && 'border-muted-foreground/40'
      )}
      disabled
    />
  )
}

export const HoistedTodoPanel: FC<{ todos: TodoItem[] }> = ({ todos }) => {
  if (!todos.length) {
    return null
  }

  const label = headerLabel(todos)

  return (
    <section
      className="mt-1 mb-3 inline-block w-fit max-w-full overflow-hidden rounded-2xl border border-border/70 bg-card align-top shadow-[0_1px_2px_0_hsl(var(--foreground)/0.04),0_1px_4px_-1px_hsl(var(--foreground)/0.06)]"
      data-slot="aui_todo-hoisted"
    >
      <header className="px-3 pt-3 pb-2">
        <span
          className="block max-w-full truncate text-[0.85rem] font-semibold leading-tight tracking-tight text-foreground"
          title={label}
        >
          {label}
        </span>
      </header>
      <ul className="grid min-w-0 gap-0.5 px-3 pb-3">
        {todos.map(todo => (
          <li
            // Active row at full presence; everything else fades. Opacity on
            // the row so the checkbox glyph dims with the text.
            className={cn(
              'flex min-w-0 items-center gap-3 py-1.5 transition-opacity',
              todo.status === 'in_progress' ? 'opacity-100' : 'opacity-45'
            )}
            key={todo.id}
          >
            <Checkmark label={todo.content} status={todo.status} />
            <span className="min-w-0 wrap-anywhere text-[0.8rem] leading-[1.2rem] text-foreground">{todo.content}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
