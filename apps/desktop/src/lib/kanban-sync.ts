import type { TodoItem, TodoStatus } from '@/lib/todos'

type KanbanStatus = 'todo' | 'ready' | 'running' | 'review' | 'done' | 'blocked'

export interface CronErrorRecord {
  jobId: string
  lastError: string
}

const TODO_TO_KANBAN_STATUS: Record<TodoStatus, string> = {
  pending: 'todo',
  in_progress: 'running',
  completed: 'done',
  cancelled: 'blocked'
}

// Track last-known cron errors to avoid re-creating blocked tasks for already-seen failures.
// Cleared on successful run (last_error becomes null/empty).
const seenCronErrors = new Map<string, string>()

/**
 * Sync agent todo status changes to mirrored Kanban tasks.
 *
 * Called after $todosBySession is updated. Finds existing kanban tasks
 * that match by externalTaskId + sessionId with syncMode = 'mirrored'
 * and updates their status accordingly.
 */
export async function syncTodoToKanbanTasks(sessionId: string, todos: TodoItem[]): Promise<void> {
  try {
    const allTasks = await window.hermesDesktop.kanban.allTasks()

    for (const todo of todos) {
      const kanbanStatus = TODO_TO_KANBAN_STATUS[todo.status]
      if (!kanbanStatus) continue

      const matchedTask = allTasks.find(
        t =>
          t.externalTaskId === todo.id &&
          t.sessionId === sessionId &&
          (t.syncMode === 'mirrored' || t.syncMode === 'linked')
      )

      if (matchedTask && matchedTask.status !== kanbanStatus) {
        await window.hermesDesktop.kanban.updateTask(matchedTask.id, {
          status: kanbanStatus as KanbanStatus,
          lastSyncedAt: Date.now()
        })
      }
    }
  } catch (e) {
    console.warn('[kanban-sync] syncTodoToKanbanTasks failed:', e)
  }
}

/**
 * Create blocked Kanban tasks for cron jobs that have new errors.
 *
 * Called during cron job polling. Tracks already-seen errors per job ID
 * to avoid creating duplicate blocked tasks.
 */
export async function syncCronFailureToKanban(jobs: Array<{ id: string; name?: string | null; last_error?: string | null }>): Promise<void> {
  try {
    const allTasks = await window.hermesDesktop.kanban.allTasks()

    for (const job of jobs) {
      const currentError = job.last_error
      const previousError = seenCronErrors.get(job.id)

      if (!currentError) {
        // Job is healthy — clear tracking so a future error is detected as new
        if (previousError !== undefined) {
          seenCronErrors.delete(job.id)
        }
        continue
      }

      // Skip if we've already seen this error
      if (currentError === previousError) continue

      // Check if a blocked task already exists for this error
      const existing = allTasks.some(
        t =>
          t.externalTaskId === job.id &&
          t.externalTaskKind === 'cron_job' &&
          t.description === currentError
      )
      if (existing) continue

      // Create blocked task
      await window.hermesDesktop.kanban.createTask({
        boardId: 'default',
        title: `Cron failed: ${job.name || job.id}`,
        description: currentError,
        status: 'blocked',
        priority: 'high',
        source: 'cron',
        externalTaskId: job.id,
        externalTaskKind: 'cron_job',
        assigneeType: 'user',
        assigneeLabel: 'You',
        syncMode: 'linked'
      })

      seenCronErrors.set(job.id, currentError)
    }
  } catch (e) {
    console.warn('[kanban-sync] syncCronFailureToKanban failed:', e)
  }
}
