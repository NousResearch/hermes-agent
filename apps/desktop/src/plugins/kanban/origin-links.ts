/**
 * Origin links — what a conversation's linked Kanban tasks are doing, as the
 * sidebar badge and the composer strip read it.
 *
 * Truth lives on the backend: `/origin-tasks` resolves the conversation's
 * lineage in the owning profile and re-reads each linked board itself, deriving
 * `activity` from the board's own rows. This module only (1) asks through the
 * query layer — batched, and only for conversations plugin REST actually
 * reaches — and (2) reduces the answer to a view. Every gap stays visible as a
 * gap: a request failure, a conversation this profile doesn't know, a ref the
 * board could not confirm and a truncated answer are all distinct from "no tasks".
 */

import { type SessionRouteContext, useQuery } from '@hermes/plugin-sdk'

import { fetchOriginTasks, originKey, routedToScope, useKanbanScope } from './api'
import type { OriginActivity, OriginRef, OriginTasksResponse } from './types'

/** What one ref (or a conversation's loudest ref) is, including "we could not tell". */
export type OriginState = 'unavailable' | OriginActivity

/** Loudest first — also the list order. */
const RANK: Record<OriginState, number> = {
  'needs-input': 0,
  background: 1,
  stale: 2,
  unavailable: 3,
  unknown: 4,
  reserved: 5,
  blocked: 6,
  review: 7,
  waiting: 8,
  queued: 9,
  done: 10,
  archived: 11
}

/** Completed work stays listed (history, navigation) but never lights a badge. */
export const isTerminalState = (state: OriginState): boolean => state === 'done' || state === 'archived'

export const refState = (ref: OriginRef): OriginState =>
  ref.evidence === 'ok' && ref.task ? ref.task.activity : 'unavailable'

export const sortOriginRefs = (refs: readonly OriginRef[]): OriginRef[] =>
  [...refs].sort((a, b) => RANK[refState(a)] - RANK[refState(b)] || b.indexed_at - a.indexed_at)

export interface OriginSummary {
  /** Refs that are not completed — what a badge counts. */
  live: number
  /** The loudest state among all refs, or null with no refs. */
  state: null | OriginState
}

export function summarizeOrigin(refs: readonly OriginRef[]): OriginSummary {
  let state: null | OriginState = null
  let live = 0

  for (const ref of refs) {
    const next = refState(ref)

    if (!isTerminalState(next)) {
      live++
    }

    if (state === null || RANK[next] < RANK[state]) {
      state = next
    }
  }

  return { live, state }
}

export type OriginView =
  | { kind: 'loading' }
  | { kind: 'out-of-scope' }
  | { kind: 'ready'; refs: OriginRef[]; truncated: null | { shown: number; total: number } }
  | { kind: 'unavailable'; reason: 'request' | 'unknown-session' }

export interface OriginQueryState {
  data: OriginTasksResponse | undefined
  status: 'error' | 'pending' | 'success'
}

/** Pure reduction of the batched answer to one conversation's view. */
export function deriveOriginView(ambient: boolean, ids: readonly string[], query: OriginQueryState): OriginView {
  if (!ambient || ids.length === 0) {
    return { kind: 'out-of-scope' }
  }

  if (query.status === 'error') {
    return { kind: 'unavailable', reason: 'request' }
  }

  const { data } = query

  if (!data) {
    return { kind: 'loading' }
  }

  // The answering profile's store knows none of this conversation's ids: it is not
  // the owner, so an empty list would be a false "no tasks".
  if (ids.every(id => data.unknown_sessions.includes(id))) {
    return { kind: 'unavailable', reason: 'unknown-session' }
  }

  const mine = new Set(ids)
  const refs = sortOriginRefs(data.refs.filter(ref => mine.has(ref.origin_session_id)))
  const incomplete = data.truncated.refs || data.truncated.lineage || data.truncated.sessions

  return {
    kind: 'ready',
    refs,
    truncated: incomplete ? { shown: refs.length, total: Math.max(data.truncated.total_refs, refs.length) } : null
  }
}

/** One conversation's origin view. Queries only when plugin REST is routed to the
 *  conversation's owner (`context.ambient`); the key carries the owner profile and
 *  every lineage id, so a compression rotation or another profile is a clean miss. */
export function useOriginView(context: SessionRouteContext): OriginView {
  const scope = useKanbanScope()
  const ids = context.lineageIds

  const query = useQuery({
    enabled: q => context.ambient && ids.length > 0 && routedToScope(q),
    queryFn: () => fetchOriginTasks(scope, context.profile, ids),
    queryKey: originKey(scope, context.profile, ids),
    staleTime: 10_000
  })

  return deriveOriginView(context.ambient, ids, { data: query.data, status: query.status })
}
