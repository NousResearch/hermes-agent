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
/** Sessions whose current list came from a durable compaction carrier. */
export const $preservedTodosBySession = atom<Record<string, true>>({})

let todoHydrationGeneration = 0
const latestTodoHydrationBySession = new Map<string, number>()
const activeTodoHydrationTokens = new Set<TodoHydrationToken>()
const activeTodoHydrationsBySession = new Map<string, Set<TodoHydrationToken>>()
const retiredTodoHydrationTokens = new WeakSet<TodoHydrationToken>()

/** @internal Read-only lifecycle accounting for bounded-retention tests. */
export function _todoHydrationAuthorityStatsForTests(): {
  activeSessionCount: number
  activeTokenCount: number
  generation: number
  latestSessionCount: number
} {
  return {
    activeSessionCount: activeTodoHydrationsBySession.size,
    activeTokenCount: activeTodoHydrationTokens.size,
    generation: todoHydrationGeneration,
    latestSessionCount: latestTodoHydrationBySession.size
  }
}

export interface TodoHydrationToken {
  /** Monotonic renderer-owned order allocated when this hydration operation starts. */
  generation: number
  /** Cold resume binds only after session.resume returns the new runtime identity. */
  runtimeSessionId?: string
}

function registerTodoHydrationForSession(token: TodoHydrationToken, runtimeSessionId: string): void {
  const activeForSession = activeTodoHydrationsBySession.get(runtimeSessionId) ?? new Set<TodoHydrationToken>()
  activeForSession.add(token)
  activeTodoHydrationsBySession.set(runtimeSessionId, activeForSession)

  if ((latestTodoHydrationBySession.get(runtimeSessionId) ?? 0) < token.generation) {
    latestTodoHydrationBySession.set(runtimeSessionId, token.generation)
  }
}

/** Begin one admitted hydration operation before its first async boundary. */
export function captureTodoWriteFence(runtimeSessionId?: string): TodoHydrationToken {
  todoHydrationGeneration += 1
  const token: TodoHydrationToken = {
    generation: todoHydrationGeneration,
    ...(runtimeSessionId ? { runtimeSessionId } : {})
  }
  activeTodoHydrationTokens.add(token)

  if (runtimeSessionId) {
    registerTodoHydrationForSession(token, runtimeSessionId)
  }

  return token
}

/** Release an admitted operation that can no longer publish. Idempotent. */
export function releaseTodoHydrationToken(token: TodoHydrationToken): void {
  if (!activeTodoHydrationTokens.delete(token)) {
    return
  }

  const runtimeSessionId = token.runtimeSessionId

  if (!runtimeSessionId) {
    return
  }

  const activeForSession = activeTodoHydrationsBySession.get(runtimeSessionId)

  if (!activeForSession) {
    return
  }

  activeForSession.delete(token)

  if (activeForSession.size === 0) {
    activeTodoHydrationsBySession.delete(runtimeSessionId)
    latestTodoHydrationBySession.delete(runtimeSessionId)
  }
}

function retireSessionTodoHydrations(runtimeSessionId: string): void {
  const activeForSession = activeTodoHydrationsBySession.get(runtimeSessionId)

  if (activeForSession) {
    for (const token of activeForSession) {
      retiredTodoHydrationTokens.add(token)
      activeTodoHydrationTokens.delete(token)
    }
  }

  activeTodoHydrationsBySession.delete(runtimeSessionId)
  latestTodoHydrationBySession.delete(runtimeSessionId)
}

function retireAllTodoHydrations(): void {
  for (const token of activeTodoHydrationTokens) {
    retiredTodoHydrationTokens.add(token)
  }

  activeTodoHydrationTokens.clear()
  activeTodoHydrationsBySession.clear()
  latestTodoHydrationBySession.clear()
  todoHydrationGeneration = 0
}

export function bindTodoHydrationToken(token: TodoHydrationToken, runtimeSessionId: string): boolean {
  if (
    !runtimeSessionId ||
    retiredTodoHydrationTokens.has(token) ||
    !activeTodoHydrationTokens.has(token) ||
    (token.runtimeSessionId && token.runtimeSessionId !== runtimeSessionId)
  ) {
    return false
  }

  if (!token.runtimeSessionId) {
    token.runtimeSessionId = runtimeSessionId
    registerTodoHydrationForSession(token, runtimeSessionId)
  }

  return latestTodoHydrationBySession.get(runtimeSessionId) === token.generation
}

function sessionTodosUnchangedSince(sid: string, token: TodoHydrationToken | undefined): boolean {
  return !token || bindTodoHydrationToken(token, sid)
}

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

export type TodoPresentationKind = 'continuing' | 'finished' | 'hidden' | 'paused' | 'restored' | 'working'

export interface TodoPresentationState {
  kind: TodoPresentationKind
  remaining: number
  stopReason?: string
}

export interface TodoPresentationInputs {
  continuation?: TodoContinuationSnapshot
  /** Durable carrier state is visible as a static plan without claiming live work. */
  preserved?: boolean
  /** Backend-confirmed turn liveness, never optimistic submit state. */
  turnLive: boolean
}

/** Resolve presentation without promoting todo row status into liveness truth. */
export function resolveTodoPresentation(
  todos: readonly TodoItem[] | null,
  { continuation, preserved, turnLive }: TodoPresentationInputs
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

  if (preserved) {
    return { kind: 'restored', remaining }
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

interface SessionTodoWriteOptions {
  forgetRevision?: boolean
  ifUnchangedSince?: TodoHydrationToken
  preserved?: boolean
  revision?: null | number
}

function removeSessionTodoState(sid: string, forgetRevision: boolean): void {
  clearTimers.cancel(sid)

  const preserved = $preservedTodosBySession.get()

  if (sid in preserved) {
    const { [sid]: _drop, ...rest } = preserved
    $preservedTodosBySession.set(rest)
  }

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

export function setSessionTodos(
  sid: string,
  todos: TodoItem[],
  optionsOrRevision: SessionTodoWriteOptions | null | number = {}
): boolean {
  const options =
    typeof optionsOrRevision === 'number' || optionsOrRevision === null
      ? { revision: optionsOrRevision }
      : optionsOrRevision
  const hydrationToken = options.ifUnchangedSince

  if (!sid || !sessionTodosUnchangedSince(sid, hydrationToken)) {
    if (hydrationToken) {
      releaseTodoHydrationToken(hydrationToken)
    }

    return false
  }

  if (!acceptRevision(sid, options.revision)) {
    if (hydrationToken) {
      releaseTodoHydrationToken(hydrationToken)
    }

    return false
  }

  if (!hydrationToken) {
    retireSessionTodoHydrations(sid)
  }

  try {
    clearTimers.cancel(sid)
    const preserved = $preservedTodosBySession.get()

    if (options.preserved) {
      $preservedTodosBySession.set({ ...preserved, [sid]: true })
    } else if (sid in preserved) {
      const { [sid]: _drop, ...rest } = preserved
      $preservedTodosBySession.set(rest)
    }

    $todosBySession.set({ ...$todosBySession.get(), [sid]: todos })

    if (!todoListActive(todos)) {
      clearTimers.schedule(sid, FINISHED_LINGER_MS, () => {
        // This delayed callback belongs to the exact rendered list above. It is
        // presentation cleanup, not newer live backend intent, so it must neither
        // clear a replacement list nor retire an in-flight hydration operation.
        if ($todosBySession.get()[sid] === todos) {
          removeSessionTodoState(sid, false)
        }
      })
    }

    return true
  } finally {
    if (hydrationToken) {
      // A current hydration publishes at most once. Once it settles, no older
      // admitted operation may become authoritative again.
      retireSessionTodoHydrations(sid)
    }
  }
}

export function clearAllSessionTodos(): void {
  for (const sid of Object.keys($todosBySession.get())) {
    clearTimers.cancel(sid)
  }

  retireAllTodoHydrations()
  $preservedTodosBySession.set({})
  $todosBySession.set({})
  $todoRevisionsBySession.set({})
}

export function clearSessionTodos(
  sid: string,
  options: Pick<SessionTodoWriteOptions, 'forgetRevision' | 'ifUnchangedSince'> = {}
): boolean {
  const hydrationToken = options.ifUnchangedSince

  if (!sid || !sessionTodosUnchangedSince(sid, hydrationToken)) {
    if (hydrationToken) {
      releaseTodoHydrationToken(hydrationToken)
    }

    return false
  }

  try {
    removeSessionTodoState(sid, options.forgetRevision !== false)
    return true
  } finally {
    retireSessionTodoHydrations(sid)
  }
}

// Drop a still-active todo list (any pending/in_progress item) at turn end when
// no authoritative controller says the plan remains active or paused. This
// prevents stale task panels while retaining durable rows that the renderer can
// truthfully show without a liveness spinner. A finished list is left untouched
// so its short linger still shows the last checkmark landing.
export function clearActiveSessionTodos(sid: string) {
  const todos = $todosBySession.get()[sid]
  const continuation = $todoContinuationsBySession.get()[sid]
  const preserved = $preservedTodosBySession.get()[sid]

  if (
    preserved ||
    !todos ||
    !todoListActive(todos) ||
    continuation?.state === 'active' ||
    continuation?.state === 'paused'
  ) {
    return
  }

  clearSessionTodos(sid, { forgetRevision: false })
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
    removeSessionTodoState(sid, false)
  }
}
