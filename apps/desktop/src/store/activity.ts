import { atom } from 'nanostores'

import type { ActionStatusResponse } from '@/types/hermes'

const HISTORY_LIMIT = 8
const COMPLETED_TTL_MS = 5 * 60 * 1000

interface DesktopActionTask {
  status: ActionStatusResponse
  updatedAt: number
}

const $desktopActionTasks = atom<Record<string, DesktopActionTask>>({})

export function upsertDesktopActionTask(status: ActionStatusResponse): void {
  $desktopActionTasks.set(prune({ ...$desktopActionTasks.get(), [status.name]: { status, updatedAt: Date.now() } }))
}

function prune(tasks: Record<string, DesktopActionTask>): Record<string, DesktopActionTask> {
  const now = Date.now()

  return Object.fromEntries(
    Object.entries(tasks)
      .filter(([, task]) => task.status.running || now - task.updatedAt <= COMPLETED_TTL_MS)
      .sort(([, left], [, right]) => right.updatedAt - left.updatedAt)
      .slice(0, HISTORY_LIMIT)
  )
}
