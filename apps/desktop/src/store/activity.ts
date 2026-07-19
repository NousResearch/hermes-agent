import { atom } from 'nanostores'

import { sessionTitle } from '@/lib/chat-runtime'
import type { PreviewServerRestart } from '@/store/preview'
import type { ActionStatusResponse, SessionInfo } from '@/types/hermes'

const HISTORY_LIMIT = 8
const COMPLETED_TTL_MS = 5 * 60 * 1000

export type RailTaskStatus = 'error' | 'running' | 'success'
export type RailTaskKind = 'action' | 'preview' | 'session'

export interface RailTask {
  detail?: string
  id: string
  kind: RailTaskKind
  label: string
  sessionId?: string
  status: RailTaskStatus
  updatedAt: number
}

export interface DesktopActionTask {
  status: ActionStatusResponse
  updatedAt: number
}

export const $desktopActionTasks = atom<Record<string, DesktopActionTask>>({})

export function upsertDesktopActionTask(status: ActionStatusResponse): void {
  $desktopActionTasks.set(prune({ ...$desktopActionTasks.get(), [status.name]: { status, updatedAt: Date.now() } }))
}

export function buildRailTasks(
  workingSessionIds: readonly string[],
  finishedSessionIds: readonly string[],
  sessions: readonly SessionInfo[],
  previewRestart: PreviewServerRestart | null,
  actionTasks: Record<string, DesktopActionTask>,
  now: number = Date.now()
): RailTask[] {
  const sessionsById = new Map(sessions.map(session => [session.id, session]))
  const working = new Set(workingSessionIds)

  const sessionTasks: RailTask[] = workingSessionIds.map((id, index) => {
    const session = sessionsById.get(id)

    return {
      id: `session:${id}`,
      kind: 'session',
      label: session ? sessionTitle(session) : 'Session task',
      sessionId: id,
      status: 'running',
      updatedAt: sessionTimestamp(session) || now - index
    }
  })

  const finishedTasks: RailTask[] = finishedSessionIds
    .filter(id => !working.has(id))
    .map((id, index) => {
      const session = sessionsById.get(id)

      return {
        id: `session:${id}`,
        kind: 'session',
        label: session ? sessionTitle(session) : 'Session task',
        sessionId: id,
        status: 'success',
        updatedAt: sessionTimestamp(session) || now - workingSessionIds.length - index
      }
    })

  const previewTasks: RailTask[] = previewRestart
    ? [
        {
          detail: previewRestart.message || previewRestart.url,
          id: `preview:${previewRestart.taskId}`,
          kind: 'preview',
          label: 'Preview restart',
          status:
            previewRestart.status === 'error' ? 'error' : previewRestart.status === 'running' ? 'running' : 'success',
          updatedAt: now
        }
      ]
    : []

  const actions: RailTask[] = Object.values(actionTasks).map(({ status, updatedAt }) => ({
    ...(status.running || status.exit_code === 0 ? {} : { detail: `Exit ${status.exit_code ?? 'unknown'}` }),
    id: `action:${status.name}`,
    kind: 'action',
    label: status.name,
    status: actionStatus(status),
    updatedAt
  }))

  return [...sessionTasks, ...finishedTasks, ...previewTasks, ...actions].sort(
    (left, right) => right.updatedAt - left.updatedAt
  )
}

const sessionTimestamp = (session: SessionInfo | undefined): number =>
  session ? (session.last_active || session.started_at) * 1000 : 0

function actionStatus(status: ActionStatusResponse): RailTaskStatus {
  if (status.running) {
    return 'running'
  }

  return status.exit_code === 0 ? 'success' : 'error'
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
