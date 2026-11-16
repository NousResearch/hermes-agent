/** Derived counts only. Checked children never complete a phase's own acceptance. */
export interface ProgressTodo {
  id: string
  parent?: string
  status: string
}

export interface TodoProgress {
  phases: { total: number; completed: number }
  subtasks: { total: number; completed: number }
  unresolved: number
}

const rooted = (id: string, parents: ReadonlyMap<string, string>): boolean => {
  const seen = new Set<string>()

  while (parents.has(id)) {
    if (seen.has(id)) {
      return false
    }

    seen.add(id)
    id = parents.get(id)!
  }

  return true
}

/** Old snapshots may have orphan or cyclic parents; keep their pending work visible. */
export function todoProgress(todos: readonly ProgressTodo[]): TodoProgress {
  const byId = new Map(todos.map(todo => [todo.id, todo]))
  const parents = new Map<string, string>()
  const active = new Set<string>()

  for (const [id, todo] of byId) {
    if (todo.parent && byId.has(todo.parent) && todo.parent !== id) {
      parents.set(id, todo.parent)
    }

    if (todo.status === 'pending' || todo.status === 'in_progress') {
      active.add(id)
    }
  }

  const unfinished = new Set(active)

  for (let id of active) {
    const seen = new Set([id])

    while (parents.has(id) && !seen.has(parents.get(id)!)) {
      id = parents.get(id)!
      seen.add(id)
      unfinished.add(id)
    }
  }

  const progress: TodoProgress = {
    phases: { total: 0, completed: 0 },
    subtasks: { total: 0, completed: 0 },
    unresolved: active.size
  }

  for (const [id, todo] of byId) {
    if (todo.status === 'cancelled') {
      continue
    }

    const counts = parents.has(id) && rooted(id, parents) ? progress.subtasks : progress.phases

    counts.total += 1
    counts.completed += Number(todo.status === 'completed' && !unfinished.has(id))
  }

  return progress
}
