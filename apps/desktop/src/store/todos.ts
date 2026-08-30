import { atom, computed } from 'nanostores'

import { keyedTimeouts } from '@/lib/keyed-timeouts'
import { stableRecord } from '@/lib/stable-array'
import { parseTodoRevision, parseTodos, type TodoItem } from '@/lib/todos'

import { $sessions, lineageAliases } from './session'
import { $sessionStates } from './session-states'

/**
 * Live todo list per runtime session, rendered by the composer status stack
 * (the inline transcript panel is gone). Fed from two places:
 *
 * - live `todo` tool events (use-message-stream)
 * - stored-session hydration (desktop-controller) — but only when the list is
 *   still in flight, so reopening an old chat doesn't pin its finished plan
 *   above the composer forever.
 */
export const $todosBySession = atom<Record<string, TodoItem[]>>({})
export const $todoRevisionsBySession = atom<Record<string, number>>({})

export const todoListActive = (todos: readonly TodoItem[]) =>
  todos.some(t => t.status === 'pending' || t.status === 'in_progress')

/**
 * Narrow renderer input for the authoritative continuation controller. The
 * goal-control backend owns these values; todo history must never manufacture
 * them. `revision` lets future async hydration reject an older snapshot.
 */
export interface TodoContinuationSnapshot {
  revision: number
  state: 'active' | 'none' | 'paused'
  stopReason?: string
}

/** Authoritative continuation cache. No renderer parser writes this store. */
export const $todoContinuationsBySession = atom<Record<string, TodoContinuationSnapshot>>({})
const continuationRevisionBySession = new Map<string, number>()

export function applyTodoContinuationSnapshot(sid: string, snapshot: TodoContinuationSnapshot): void {
  if (!sid || !Number.isFinite(snapshot.revision)) {
    return
  }

  const previousRevision = continuationRevisionBySession.get(sid)

  if (previousRevision !== undefined && snapshot.revision < previousRevision) {
    return
  }

  continuationRevisionBySession.set(sid, snapshot.revision)
  const current = $todoContinuationsBySession.get()

  if (snapshot.state === 'none') {
    if (sid in current) {
      const { [sid]: _drop, ...rest } = current
      $todoContinuationsBySession.set(rest)
    }

    const session = $sessionStates.get()[sid]
    const todos = $todosBySession.get()[sid]

    if (todos && todoListActive(todos) && !(session?.busy && session.turnLive)) {
      clearSessionTodos(sid)
    }

    return
  }

  $todoContinuationsBySession.set({ ...current, [sid]: snapshot })
}

/** Retire one controller scope without treating reset as an authoritative `none`. */
export function clearTodoContinuation(sid: string): void {
  continuationRevisionBySession.delete(sid)
  const current = $todoContinuationsBySession.get()

  if (sid in current) {
    const { [sid]: _drop, ...rest } = current
    $todoContinuationsBySession.set(rest)
  }
}

/** Clear controller snapshots and revision fences when their gateway scope is retired. */
export function clearAllTodoContinuations(): void {
  continuationRevisionBySession.clear()
  $todoContinuationsBySession.set({})
}

export type TodoPresentationKind = 'continuing' | 'finished' | 'hidden' | 'paused' | 'working'

export interface TodoPresentationState {
  kind: TodoPresentationKind
  remaining: number
  stopReason?: string
}

export interface TodoPresentationInputs {
  continuation?: TodoContinuationSnapshot
  /** Backend-confirmed turn liveness, never optimistic submit state. */
  turnLive: boolean
}

/** Resolve presentation without promoting todo row status into liveness truth. */
export function resolveTodoPresentation(
  todos: readonly TodoItem[] | null,
  { continuation, turnLive }: TodoPresentationInputs
): TodoPresentationState {
  const remaining = todos?.filter(t => t.status === 'pending' || t.status === 'in_progress').length ?? 0

  if (!todos || todos.length === 0) {
    return { kind: 'hidden', remaining: 0 }
  }

  if (remaining === 0) {
    return { kind: 'finished', remaining: 0 }
  }

  if (turnLive) {
    return { kind: 'working', remaining }
  }

  if (continuation?.state === 'active') {
    return { kind: 'continuing', remaining }
  }

  if (continuation?.state === 'paused') {
    return { kind: 'paused', remaining, ...(continuation.stopReason ? { stopReason: continuation.stopReason } : {}) }
  }

  return { kind: 'hidden', remaining }
}

let todoProgress: Readonly<Record<string, string>> = {}

/** Live "X/Y" per STORED session id, for the sidebar's inbox cards. The live
 *  map keys on runtime ids; this projects through the same storedSessionId +
 *  lineage-alias fallback as the working/attention projections, so the card
 *  finds its count under the id the sidebar knows. Cancelled items don't
 *  count toward either side of the fraction. Values are the rendered "X/Y"
 *  string — primitives, so stableRecord can suppress no-op emits. */
export const $todoProgressBySession = computed(
  [$todosBySession, $sessionStates, $sessions],
  (todosMap, states, sessions) => {
    const next: Record<string, string> = {}

    for (const [runtimeId, todos] of Object.entries(todosMap)) {
      const counted = todos.filter(t => t.status !== 'cancelled')

      if (counted.length === 0) {
        continue
      }

      const progress = `${counted.filter(t => t.status === 'completed').length}/${counted.length}`

      for (const alias of lineageAliases(states[runtimeId]?.storedSessionId ?? runtimeId, sessions)) {
        next[alias] = progress
      }
    }

    return (todoProgress = stableRecord(todoProgress, next))
  }
)

// Decide which todo list to restore when rehydrating a session from stored
// history. Without authoritative continuation, rehydration after a completed
// turn treats an active list as stale so it cannot undo turn-end cleanup. A
// future typed goal snapshot may explicitly restore unfinished rows as static
// continuing/paused work. Finished rows keep their existing short linger.
// Returns null when there is nothing safe to restore.
export function todosForHydration(
  todos: readonly TodoItem[] | null,
  continuation?: TodoContinuationSnapshot
): TodoItem[] | null {
  if (!todos) {
    return null
  }

  if (!todoListActive(todos) || continuation?.state === 'active' || continuation?.state === 'paused') {
    return [...todos]
  }

  return null
}

// Once a list finishes (every item completed/cancelled), the final state
// lingers just long enough to see the last checkmark land, then the group
// drops out of the stack on its own.
const FINISHED_LINGER_MS = 4_000
const clearTimers = keyedTimeouts()

function acceptRevision(sid: string, revision?: null | number): boolean {
  const revisions = $todoRevisionsBySession.get()
  const current = revisions[sid]

  // tool.start has no revision. Apply the merge locally and leave the
  // watermark alone so a later todo.updated / tool.complete can still win.
  if (revision == null) {
    return true
  }

  if (current != null && revision < current) {
    return false
  }

  if (current !== revision) {
    $todoRevisionsBySession.set({ ...revisions, [sid]: revision })
  }

  return true
}

export function setSessionTodos(sid: string, todos: TodoItem[], revision?: null | number) {
  if (!sid) {
    return
  }

  if (!acceptRevision(sid, revision)) {
    return
  }

  clearTimers.cancel(sid)
  $todosBySession.set({ ...$todosBySession.get(), [sid]: todos })

  if (!todoListActive(todos)) {
    clearTimers.schedule(sid, FINISHED_LINGER_MS, () => dropSessionTodos(sid, false))
  }
}

function dropSessionTodos(sid: string, forgetRevision: boolean) {
  clearTimers.cancel(sid)

  const map = $todosBySession.get()

  if (sid in map) {
    const { [sid]: _drop, ...rest } = map
    $todosBySession.set(rest)
  }

  if (forgetRevision) {
    const revisions = $todoRevisionsBySession.get()

    if (sid in revisions) {
      const { [sid]: _drop, ...rest } = revisions
      $todoRevisionsBySession.set(rest)
    }
  }
}

export function clearAllSessionTodos(): void {
  for (const sid of Object.keys($todosBySession.get())) {
    clearTimers.cancel(sid)
  }

  $todosBySession.set({})
  $todoRevisionsBySession.set({})
}

export function clearSessionTodos(sid: string) {
  dropSessionTodos(sid, true)
}

// Drop a still-active todo list (any pending/in_progress item) at turn end when
// no authoritative controller says the plan remains active or paused. This
// prevents stale task panels while retaining durable rows that the renderer can
// truthfully show without a liveness spinner. A finished list is left untouched
// so its short linger still shows the last checkmark landing.
export function clearActiveSessionTodos(sid: string) {
  const todos = $todosBySession.get()[sid]
  const continuation = $todoContinuationsBySession.get()[sid]

  if (!todos || !todoListActive(todos) || continuation?.state === 'active' || continuation?.state === 'paused') {
    return
  }

  dropSessionTodos(sid, false)
}

/** Apply a session.resume/activate or todo.updated full snapshot. Idle
 * sessions keep the existing stale-active guard; running sessions restore the
 * active plan because the backend has proved that turn is still live. */
export function restoreSessionTodosFromSnapshot(sid: string, snapshot: unknown, running: boolean) {
  const todos = parseTodos(snapshot)

  if (!sid || todos === null) {
    return
  }

  const revision = parseTodoRevision(snapshot)

  // An unused store serializes as {todos: [], revision: 0}. That is not a
  // real snapshot. Applying it would stamp watermark 0 and leave an empty
  // list in the map.
  if (todos.length === 0 && (revision == null || revision === 0)) {
    return
  }

  const visible = running ? todos : todosForHydration(todos)

  if (visible !== null) {
    setSessionTodos(sid, visible, revision)
  } else if (acceptRevision(sid, revision)) {
    dropSessionTodos(sid, false)
  }
}
