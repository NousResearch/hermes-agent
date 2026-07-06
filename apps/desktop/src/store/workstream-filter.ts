import { atom } from 'nanostores'

import { type Codec, Codecs, persistentAtom } from '@/lib/persisted'
import type { TodoItem } from '@/lib/todos'
import type { SubagentProgress } from '@/store/subagents'
import { deriveWorkstreamActivity, liveWorkstreamSessionId, type WorkstreamActivity } from '@/store/workstream'
import { workstreamLifecycle, type WorkstreamLifecycle, type WorkstreamMetadataBySession } from '@/store/workstream-metadata'
import type { SessionInfo } from '@/types/hermes'

const WORKSTREAM_FILTER_KEY = 'hermes.desktop.workstream-filter.v1'

export const WORKSTREAM_FILTERS = ['all', 'active', 'blocked', 'review', 'closed', 'safe-delete'] as const
export type WorkstreamFilter = (typeof WORKSTREAM_FILTERS)[number]

const FILTER_SET: ReadonlySet<string> = new Set(WORKSTREAM_FILTERS)
const BLOCKED_STATES = new Set<WorkstreamActivity['state']>(['blocked', 'restart', 'warn'])
const REVIEW_STATES = new Set<WorkstreamActivity['state']>(['plan_review', 'verify'])

const filterCodec: Codec<WorkstreamFilter> = {
  decode: raw => (FILTER_SET.has(raw) ? (raw as WorkstreamFilter) : 'all'),
  encode: value => (value === 'all' ? null : Codecs.text.encode(value))
}

export const $workstreamFilter = persistentAtom<WorkstreamFilter>(WORKSTREAM_FILTER_KEY, 'all', filterCodec)
export const $workstreamVisibleSessionIds = atom<string[]>([])

export interface WorkstreamFilterRuntime {
  activeSessionId: null | string
  attentionSessionIds: readonly string[]
  metadataBySession: WorkstreamMetadataBySession
  selectedStoredSessionId: null | string
  subagentsBySession: Record<string, readonly SubagentProgress[] | undefined>
  todosBySession: Record<string, readonly TodoItem[] | undefined>
  workingSessionIds: readonly string[]
}

function sessionValue<T>(bySession: Record<string, readonly T[] | undefined>, sessionId: string, liveSessionId: string) {
  return liveSessionId === sessionId ? bySession[sessionId] : (bySession[liveSessionId] ?? bySession[sessionId])
}

export function workstreamFilterPredicate(
  filter: WorkstreamFilter,
  activity: WorkstreamActivity,
  lifecycle: WorkstreamLifecycle
): boolean {
  if (filter === 'all') {
    return true
  }

  if (filter === 'closed') {
    return lifecycle === 'closed'
  }

  if (filter === 'safe-delete') {
    return lifecycle === 'safe_delete'
  }

  if (lifecycle !== 'active' && lifecycle !== 'restart_required') {
    return false
  }

  if (filter === 'blocked') {
    return lifecycle === 'restart_required' || BLOCKED_STATES.has(activity.state)
  }

  if (filter === 'review') {
    return lifecycle === 'active' && REVIEW_STATES.has(activity.state)
  }

  return lifecycle === 'active' && !BLOCKED_STATES.has(activity.state) && !REVIEW_STATES.has(activity.state)
}

export function workstreamActivityForSession(
  session: Pick<SessionInfo, 'id'>,
  runtime: WorkstreamFilterRuntime
): { activity: WorkstreamActivity; lifecycle: WorkstreamLifecycle } {
  const liveSessionId = liveWorkstreamSessionId(session.id, runtime.activeSessionId, runtime.selectedStoredSessionId)
  const sessionIds = liveSessionId === session.id ? [session.id] : [session.id, liveSessionId]
  const lifecycle = workstreamLifecycle(session.id, runtime.metadataBySession)

  const activity = deriveWorkstreamActivity({
    explicitState: lifecycle === 'restart_required' ? 'restart' : lifecycle === 'closed' || lifecycle === 'safe_delete' ? 'close' : null,
    isWorking: sessionIds.some(id => runtime.workingSessionIds.includes(id)),
    needsInput: sessionIds.some(id => runtime.attentionSessionIds.includes(id)),
    session: { id: session.id },
    subagents: sessionValue(runtime.subagentsBySession, session.id, liveSessionId),
    todos: sessionValue(runtime.todosBySession, session.id, liveSessionId)
  })

  return { activity, lifecycle }
}

export function filterSessionsByWorkstream(
  sessions: readonly SessionInfo[],
  filter: WorkstreamFilter,
  runtime: WorkstreamFilterRuntime
): SessionInfo[] {
  if (filter === 'all') {
    return [...sessions]
  }

  return sessions.filter(session => {
    const { activity, lifecycle } = workstreamActivityForSession(session, runtime)

    return workstreamFilterPredicate(filter, activity, lifecycle)
  })
}

export function displaySessionsForWorkstreamFilter(
  sessions: readonly SessionInfo[],
  visibleCount: number,
  filter: WorkstreamFilter
): SessionInfo[] {
  if (filter === 'all') {
    return sessions.slice(0, visibleCount)
  }

  return [...sessions]
}

export function collectWorkstreamVisibleSessionIds(
  sections: readonly (readonly SessionInfo[] | undefined)[],
  filter: WorkstreamFilter,
  runtime: WorkstreamFilterRuntime
): string[] {
  const ids: string[] = []
  const seen = new Set<string>()

  for (const sessions of sections) {
    for (const session of filterSessionsByWorkstream(sessions ?? [], filter, runtime)) {
      if (!seen.has(session.id)) {
        seen.add(session.id)
        ids.push(session.id)
      }
    }
  }

  return ids
}

export const WORKSTREAM_SESSION_ROW_SELECTOR = '[data-workstream-session-id]'

export function collectRenderedWorkstreamSessionIds(root: ParentNode | null): string[] {
  if (!root) {
    return []
  }

  const ids: string[] = []
  const seen = new Set<string>()

  for (const row of root.querySelectorAll<HTMLElement>(WORKSTREAM_SESSION_ROW_SELECTOR)) {
    const id = row.dataset.workstreamSessionId?.trim()

    if (id && !seen.has(id)) {
      seen.add(id)
      ids.push(id)
    }
  }

  return ids
}

export function setWorkstreamFilter(filter: WorkstreamFilter): void {
  $workstreamFilter.set(filter)
}

export function cycleWorkstreamFilter(): WorkstreamFilter {
  const current = $workstreamFilter.get()
  const currentIndex = WORKSTREAM_FILTERS.indexOf(current)
  const next = WORKSTREAM_FILTERS[(currentIndex + 1) % WORKSTREAM_FILTERS.length] ?? 'all'
  $workstreamFilter.set(next)

  return next
}

export function setWorkstreamVisibleSessionIds(ids: readonly string[]): void {
  const current = $workstreamVisibleSessionIds.get()

  if (current.length === ids.length && current.every((id, index) => id === ids[index])) {
    return
  }

  $workstreamVisibleSessionIds.set([...ids])
}

export function adjacentWorkstreamSessionId(currentSessionId: null | string, direction: 1 | -1): null | string {
  const ids = $workstreamVisibleSessionIds.get()

  if (ids.length === 0) {
    return null
  }

  const currentIndex = currentSessionId ? ids.indexOf(currentSessionId) : -1
  const start = currentIndex === -1 ? (direction === 1 ? -1 : 0) : currentIndex
  const nextIndex = ((start + direction) % ids.length + ids.length) % ids.length

  return ids[nextIndex] ?? null
}
