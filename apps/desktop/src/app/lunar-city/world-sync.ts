import type { AgentNoticePayload } from '@/store/agent-notices'
import { type AgentNoticeListener, subscribeAgentNotices } from '@/store/agent-notices'
import {
  $worldCursors,
  recordWorldCursor as persistWorldCursor,
  setWorldOpenedAt,
  setWorldProjection,
  type WorldCursorState,
  type WorldProjection
} from '@/store/lunar-city'

import {
  type CompletionEvent,
  type KanbanEventsListener,
  subscribeKanbanEvents
} from '../../plugins/kanban/completion-notify'
import type { KanbanTask } from '../../plugins/kanban/types'

import {
  classifyTaskCondition,
  dedupeWorldEvents,
  type ExternalWorldEventInput,
  normalizeAgentNotice,
  normalizeExternalEvent,
  normalizeKanbanEvent,
  type WorldCondition,
  type WorldEvent,
  type WorldSource
} from './world-events'

export interface WorldSnapshot {
  externalEvents?: ExternalWorldEventInput[]
  tasks: KanbanTask[]
}

export interface WorldSourceDoors {
  external?: (listener: (input: ExternalWorldEventInput) => void) => () => void
  kanban?: (listener: KanbanEventsListener) => () => void
  notices?: (listener: AgentNoticeListener) => () => void
}

export interface WorldSyncSink {
  getCursors: () => WorldCursorState
  publish: (projection: WorldProjection, cursors: WorldCursorState) => void
}

export interface ReconciledWorldState {
  cursors: WorldCursorState
  projection: WorldProjection
}

const MAX_RECENT_EVENTS = 50
const MAX_TRANSITIONS_PER_REOPEN = 12

function sourceScope(source: WorldSource, board?: string): string {
  return `${source}:${board ?? 'global'}`
}

export function worldSourceScopeKey(event: Pick<WorldEvent, 'source' | 'sourceRef'>): string {
  return sourceScope(event.source, event.sourceRef?.board)
}

function numericId(id: string): number | null {
  const value = Number(id.slice(id.lastIndexOf(':') + 1))

  return Number.isSafeInteger(value) ? value : null
}

function wasSeen(event: WorldEvent, cursors: WorldCursorState): boolean {
  const cursor = cursors.bySource[worldSourceScopeKey(event)]

  if (!cursor) {
    return false
  }

  if (cursor === event.id) {
    return true
  }

  const cursorNumber = numericId(cursor)
  const eventNumber = numericId(event.id)

  return cursorNumber !== null && eventNumber !== null && eventNumber <= cursorNumber
}

function updateCursors(events: readonly WorldEvent[], cursors: WorldCursorState): WorldCursorState {
  const next = { ...cursors, bySource: { ...cursors.bySource } }

  for (const event of events) {
    const scope = worldSourceScopeKey(event)
    const prior = next.bySource[scope]
    const priorNumber = prior ? numericId(prior) : null
    const eventNumber = numericId(event.id)

    if (!prior || (priorNumber !== null && eventNumber !== null && eventNumber > priorNumber)) {
      next.bySource[scope] = event.id
    }
  }

  return next
}

function emptyProjection(): WorldProjection {
  return { conditions: [], recentEvents: [], sourceError: null, stale: false, transitions: [] }
}

export function reconcileWorldSnapshot(
  snapshot: WorldSnapshot,
  incoming: readonly WorldEvent[],
  cursors: WorldCursorState,
  now = Date.now()
): ReconciledWorldState {
  const conditions = snapshot.tasks
    .map(task => classifyTaskCondition(task, now))
    .filter((condition): condition is WorldCondition => condition !== null)

  const snapshotEvents = (snapshot.externalEvents ?? []).map(input => normalizeExternalEvent(input, now))
  const allEvents = dedupeWorldEvents([], [...incoming, ...snapshotEvents])
  const transitions = allEvents.filter(event => !wasSeen(event, cursors)).slice(-MAX_TRANSITIONS_PER_REOPEN)
  const nextCursors = updateCursors(allEvents, cursors)

  const projection: WorldProjection = {
    conditions,
    recentEvents: allEvents.slice(-MAX_RECENT_EVENTS),
    sourceError: null,
    stale: false,
    transitions
  }

  return { cursors: { ...nextCursors, lastOpenedAt: now }, projection }
}

export function bindWorldSources(doors: WorldSourceDoors, sink: WorldSyncSink): () => void {
  const unsubs: Array<() => void> = []

  const publishEvents = (events: WorldEvent[]) => {
    if (events.length === 0) {
      return
    }

    const current = sink.getCursors()
    const transitions = events.filter(event => !wasSeen(event, current))

    const projection = {
      ...emptyProjection(),
      recentEvents: events.slice(-MAX_RECENT_EVENTS),
      transitions: transitions.slice(-MAX_TRANSITIONS_PER_REOPEN)
    }

    const cursors = updateCursors(events, current)

    sink.publish(projection, { ...cursors, lastOpenedAt: Date.now() })
  }

  const onKanban: KanbanEventsListener = (board, events: CompletionEvent[]) => {
    publishEvents(events.map(event => normalizeKanbanEvent(board, event)))
  }

  const onNotice: AgentNoticeListener = (payload: AgentNoticePayload) => {
    const event = normalizeAgentNotice(payload)

    if (event) {
      publishEvents([event])
    }
  }

  unsubs.push((doors.kanban ?? subscribeKanbanEvents)(onKanban))
  unsubs.push((doors.notices ?? subscribeAgentNotices)(onNotice))

  if (doors.external) {
    unsubs.push(doors.external(input => publishEvents([normalizeExternalEvent(input)])))
  }

  setWorldOpenedAt()

  return () => {
    unsubs.forEach(unsubscribe => unsubscribe())
  }
}

export async function refreshWorldProjection(
  snapshot: () => Promise<WorldSnapshot>,
  sink: WorldSyncSink
): Promise<void> {
  try {
    const result = reconcileWorldSnapshot(await snapshot(), [], sink.getCursors())

    sink.publish(result.projection, result.cursors)
  } catch (error) {
    const prior = sink.getCursors()
    sink.publish(
      { ...emptyProjection(), sourceError: error instanceof Error ? error.message : String(error), stale: true },
      prior
    )
  }
}

export function storeWorldSyncSink(): WorldSyncSink {
  return {
    getCursors: () => $worldCursors.get(),
    publish: (projection, cursors) => {
      setWorldProjection(projection)

      if (cursors.lastOpenedAt !== null) {
        setWorldOpenedAt(cursors.lastOpenedAt)
      }

      for (const [scope, id] of Object.entries(cursors.bySource)) {
        persistWorldCursor(scope, id)
      }
    }
  }
}
