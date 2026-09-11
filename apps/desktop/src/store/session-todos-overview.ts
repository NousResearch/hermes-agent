import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'
import { parseTodos, type TodoItem } from '@/lib/todos'

import { $gateway, activeGatewayConnectionId, activeGatewayProfileKey } from './gateway'
import { notifyError } from './notifications'
import { $sessions, getSessionOwnerHint, lineageAliases, setSessionOwnerHint } from './session'
import { ambientRequestFor } from './session-gone-latch'
import { $sessionStates, requestForOwnedSession } from './session-states'
import { $rawTodosBySession, todoListActive } from './todos'

/**
 * All-sessions todo overview — the data source for the Agents panel's "할일
 * 현황" (Task overview) section. Sourced from `$rawTodosBySession` in
 * store/todos.ts: an UNFILTERED per-runtime-session todo feed, distinct from
 * `$todosBySession` / `$todoProgressBySession`, which are composer-chip-
 * oriented and intentionally drop a stale active list on hydrate (a dead
 * turn's plan must not pin the "Tasks N/M" chip above the composer forever).
 * This overview has the opposite requirement — an idle session with pending
 * items is exactly what a persistent task list should keep showing — so it
 * must not inherit that filter.
 *
 * This store instead keeps the LAST todo snapshot seen for every stored
 * session, PERSISTED to localStorage so it survives an app restart — the
 * overview grows organically as you work and is not lost on relaunch, unlike
 * the (deliberately ephemeral) subagent tree panel. It still does NOT
 * backfill history for sessions that were never opened/resumed in this or a
 * past window (no new backend calls, no per-session transcript re-fetch): a
 * session enters the overview once its own snapshot has actually been
 * observed at least once, via a live todo.updated event OR a session.resume/
 * activate response — the latter now included specifically so reopening an
 * idle chat with pending todos surfaces them here immediately.
 */
export interface SessionTodoOverviewRow {
  storedSessionId: string
  title: string
  todos: TodoItem[]
  updatedAt: number
}

type StoredSnapshot = { todos: TodoItem[]; updatedAt: number }

// A finished (all completed/cancelled) row auto-prunes from the overview and
// its localStorage backing store after this long untouched — the store has
// no other size cap, and a stale "할일 현황" full of weeks-old finished work
// buries what's actually in flight. Chosen generously: long enough that a
// week-long trip doesn't lose yesterday's finished checklist, short enough
// that the store doesn't grow forever for an active daily user.
const FINISHED_ROW_RETENTION_MS = 7 * 24 * 60 * 60 * 1000

const SNAPSHOTS_KEY = 'hermes.desktop.taskOverview.snapshots'
const DISMISSED_KEY = 'hermes.desktop.taskOverview.dismissed'
const DISMISSED_ITEMS_KEY = 'hermes.desktop.taskOverview.dismissedItems'

function sanitizeSnapshots(value: unknown): Record<string, StoredSnapshot> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const out: Record<string, StoredSnapshot> = {}

  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (!raw || typeof raw !== 'object') {
      continue
    }

    const todos = (raw as { todos?: unknown }).todos
    const updatedAt = (raw as { updatedAt?: unknown }).updatedAt

    if (Array.isArray(todos) && typeof updatedAt === 'number') {
      out[key] = { todos: todos as TodoItem[], updatedAt }
    }
  }

  return out
}

function sanitizeStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []
}

function sanitizeDismissedItems(value: unknown): Record<string, string[]> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const out: Record<string, string[]> = {}

  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    const ids = sanitizeStringArray(raw)

    if (ids.length > 0) {
      out[key] = ids
    }
  }

  return out
}

// storedSessionId -> last-seen full todo snapshot + a wall-clock stamp, so
// the panel can sort "most recently touched first" without re-deriving it
// from the todos array itself (which carries no timestamp). Seeded from
// localStorage so a relaunch resumes with the last-known state instead of a
// blank slate; every mutation below calls `persistSnapshots()` to keep the
// disk copy in step.
const snapshots = new Map<string, StoredSnapshot>(Object.entries(sanitizeSnapshots(readJson(SNAPSHOTS_KEY))))

function persistSnapshots(): void {
  writeJson(SNAPSHOTS_KEY, Object.fromEntries(snapshots))
}

// Session ids the user has dismissed from THIS overview panel. Purely a
// display-layer hide: the underlying session/todo data is untouched, so
// reopening that chat (or a fresh todo.updated event landing on it) is
// unaffected. PERSISTED so a dismissed row does not resurface just because
// the app restarted — the user's "get this off my list" choice sticks until
// they explicitly reopen that session or a new todo.updated event resurrects
// it (see `captureSnapshot`, which still checks this set on every capture).
const dismissed = new Set<string>(sanitizeStringArray(readJson(DISMISSED_KEY)))

function persistDismissed(): void {
  writeJson(DISMISSED_KEY, dismissed.size === 0 ? null : [...dismissed])
}

// Individual todo items the user has dismissed FROM A ROW's checklist, keyed
// by storedSessionId -> set of item ids. Same display-only, persisted
// contract as `dismissed` above: there is no backend RPC to delete one item
// out of a session's todo list (the tool always rewrites the WHOLE list), so
// this filters the row's rendered `todos` array in `sessionTodoOverviewRows()`
// rather than mutating the source. A session that is still actively running
// can still resurrect a dismissed id on its next `todo.updated` snapshot
// (the same way `dismissed` above can be reopened by a fresh event) — that
// is expected: dismiss is "get this off my screen", not "delete forever" for
// a list the agent may still be updating.
const dismissedItems = new Map<string, Set<string>>(
  Object.entries(sanitizeDismissedItems(readJson(DISMISSED_ITEMS_KEY))).map(([id, ids]) => [id, new Set(ids)])
)

function persistDismissedItems(): void {
  const plain = Object.fromEntries([...dismissedItems].map(([id, ids]) => [id, [...ids]]))

  writeJson(DISMISSED_ITEMS_KEY, dismissedItems.size === 0 ? null : plain)
}

function sessionTitleFor(storedSessionId: string): string {
  const row = $sessions.get().find(s => s.id === storedSessionId)

  return row?.title?.trim() || row?.preview?.trim() || storedSessionId
}

function computeRows(): SessionTodoOverviewRow[] {
  const rows: SessionTodoOverviewRow[] = []
  const now = Date.now()
  let pruned = false

  for (const [storedSessionId, snapshot] of [...snapshots]) {
    if (dismissed.has(storedSessionId)) {
      // Belt-and-suspenders: normal invariant is that dismissSessionTodoOverviewRow
      // already deleted this id from `snapshots`, and captureSnapshot/backfill never
      // re-add an id while it's still dismissed. If this branch ever fires it means
      // something upstream drifted — clean it up AND persist so localStorage doesn't
      // silently disagree with the in-memory map.
      snapshots.delete(storedSessionId)
      pruned = true

      continue
    }

    const { todos, updatedAt } = snapshot

    // Auto-prune a FINISHED list (every item completed/cancelled) once it's
    // sat untouched past the retention window — otherwise this store only
    // ever grows (localStorage-persisted, no session-count cap, no TTL) and
    // "할일 현황" ends up showing work that finished weeks ago right next to
    // what's actually in flight. A still-active list (any pending/in_progress
    // item) is exempt no matter its age — that's real outstanding work, not
    // stale history. Backfilled rows stamp updatedAt: 0, so they're eligible
    // immediately once finished (the backend has no capture time to give us).
    if (!todoListActive(todos) && now - updatedAt > FINISHED_ROW_RETENTION_MS) {
      snapshots.delete(storedSessionId)
      pruned = true

      continue
    }

    const hiddenItems = dismissedItems.get(storedSessionId)
    const visibleTodos = hiddenItems ? todos.filter(item => !hiddenItems.has(item.id)) : todos

    rows.push({ storedSessionId, title: sessionTitleFor(storedSessionId), todos: visibleTodos, updatedAt })
  }

  if (pruned) {
    persistSnapshots()
  }

  return rows.sort((a, b) => b.updatedAt - a.updatedAt)
}

// The rows array itself lives in the atom (not a side "tick" counter a
// consumer must react to indirectly by re-deriving data through a plain
// function call). A component does `useStore($sessionTodoOverviewRows)` and
// gets the real snapshot directly — the canonical nanostores subscription
// shape, and immune to the class of bug where a component subscribes to an
// unrelated counter atom that changes correctly (proven by an independent
// `.listen()` firing) while React never repaints because the actual payload
// it reads is fetched through a second, unsynchronized hop. That exact
// failure mode was reproduced live: the panel's data (queried directly)
// was always current, but an already-mounted panel kept painting its stale
// empty state until the user closed and reopened it (forcing a remount that
// re-reads fresh). Recomputed and republished every time the underlying
// data can change, so `useStore` is the only wiring a consumer needs.
// Seeded synchronously from the persisted snapshots read above, so a
// relaunch shows the last-known list before the first live event arrives.
export const $sessionTodoOverviewRows = atom<SessionTodoOverviewRow[]>(computeRows())

function publishRows(): void {
  $sessionTodoOverviewRows.set(computeRows())
}

function captureSnapshot(): void {
  const todosByRuntime = $rawTodosBySession.get()
  const states = $sessionStates.get()
  const sessions = $sessions.get()
  let dismissedChanged = false

  for (const [runtimeId, todos] of Object.entries(todosByRuntime)) {
    if (todos.length === 0) {
      continue
    }

    const storedId = states[runtimeId]?.storedSessionId ?? runtimeId

    for (const alias of lineageAliases(storedId, sessions)) {
      // A dismissed row stays hidden from FURTHER stale/no-op captures of the
      // same finished list — but "get this off my screen" must not become a
      // permanent block on that session. New real activity (any pending/
      // in_progress item) is a fresh event worth surfacing, exactly like the
      // module doc promises ("a fresh todo.updated event landing on it" un-
      // dismisses). Without this check, `dismissed` never sheds an id once
      // added and the row can never come back no matter what the agent does
      // in that session afterward.
      if (dismissed.has(alias)) {
        if (!todoListActive(todos)) {
          continue
        }

        dismissed.delete(alias)
        dismissedChanged = true
      }

      snapshots.set(alias, { todos, updatedAt: Date.now() })
    }
  }

  if (dismissedChanged) {
    persistDismissed()
  }

  persistSnapshots()
  publishRows()
}

// Subscribed unconditionally at module load — runs for the lifetime of the
// window regardless of whether any component is currently reading the
// overview (i.e. whether the Agents panel is open). `.subscribe` (not
// `.listen`) fires once immediately too, seeding any todos already in flight
// when this module first evaluates.
$rawTodosBySession.subscribe(captureSnapshot)

/** Legacy snapshot read — still handy for one-off reads outside a React
 *  render (e.g. diagnostics), but a component should prefer
 *  `useStore($sessionTodoOverviewRows)` so it repaints on every change. */
export function sessionTodoOverviewRows(): SessionTodoOverviewRow[] {
  return $sessionTodoOverviewRows.get()
}

/** Hide one todo ITEM from a row's checklist (display-only, persisted — see
 *  `dismissedItems` above). Idempotent; republishes immediately so any
 *  subscriber of `$sessionTodoOverviewRows` repaints without a manual tick. */
export function dismissTodoOverviewItem(storedSessionId: string, itemId: string): void {
  const existing = dismissedItems.get(storedSessionId)

  if (existing) {
    existing.add(itemId)
  } else {
    dismissedItems.set(storedSessionId, new Set([itemId]))
  }

  persistDismissedItems()
  publishRows()
}

/** Actually cancel a still-active (pending/in_progress) todo item — unlike
 *  `dismissTodoOverviewItem` above, this reaches the LIVE session's authoritative
 *  TodoStore via `todo.cancel_item` (tui_gateway/methods_todo.py) so the agent
 *  itself is steered away from the task, not just hidden from this window.
 *  Optimistic: the local snapshot flips to 'cancelled' immediately so the row
 *  repaints without waiting on the round trip; a failure rolls the item back
 *  to its prior status and surfaces the error (root AGENTS.md: "Be optimistic,
 *  then honest"). The next `todo.updated` event (or this call's own success)
 *  is the authoritative correction either way. */
export async function cancelTodoOverviewItem(storedSessionId: string, itemId: string): Promise<void> {
  const snapshot = snapshots.get(storedSessionId)
  const item = snapshot?.todos.find(candidate => candidate.id === itemId)

  if (!snapshot || !item) {
    return
  }

  const previousStatus = item.status

  item.status = 'cancelled'
  publishRows()

  const gateway = $gateway.get()

  if (!gateway) {
    item.status = previousStatus
    publishRows()
    notifyError(new Error('Gateway is not connected'), 'Could not cancel the task')

    return
  }

  try {
    await requestForOwnedSession(storedSessionId, ambientRequestFor(gateway), 'todo.cancel_item', {
      stored_session_id: storedSessionId,
      item_id: itemId
    })
    persistSnapshots()
  } catch (err) {
    item.status = previousStatus
    publishRows()
    notifyError(err, 'Could not cancel the task')
  }
}

/** Move one todo item from one session's list to another's (desktop Task Overview drag-and-
 *  drop). Both sessions must be LIVE (an active AIAgent, not just a stored row this window has
 *  a cold snapshot of) — see `todo.move_item` (tui_gateway/methods_todo.py) for why: the
 *  TodoStore is in-memory per agent, so there's nothing to move out of/into for a session
 *  whose agent has been torn down. Optimistic on the SOURCE row for immediate feedback (moved
 *  item vanishes from the row being dragged from); the destination row is left to arrive via
 *  its own `todo.updated` event rather than fabricated locally, since the destination may
 *  rename the id to avoid a collision and the true assigned id is only known after the RPC
 *  returns. A failure rolls the source back and surfaces the error. */
export async function moveTodoOverviewItem(
  fromStoredSessionId: string,
  toStoredSessionId: string,
  itemId: string
): Promise<void> {
  if (fromStoredSessionId === toStoredSessionId) {
    return
  }

  const fromSnapshot = snapshots.get(fromStoredSessionId)
  const item = fromSnapshot?.todos.find(candidate => candidate.id === itemId)

  if (!fromSnapshot || !item) {
    return
  }

  const previousTodos = fromSnapshot.todos
  fromSnapshot.todos = previousTodos.filter(candidate => candidate.id !== itemId && candidate.parent !== itemId)
  publishRows()

  const gateway = $gateway.get()

  if (!gateway) {
    fromSnapshot.todos = previousTodos
    publishRows()
    notifyError(new Error('Gateway is not connected'), 'Could not move the task')

    return
  }

  try {
    await requestForOwnedSession(fromStoredSessionId, ambientRequestFor(gateway), 'todo.move_item', {
      from_session_id: fromStoredSessionId,
      to_session_id: toStoredSessionId,
      item_id: itemId
    })
    persistSnapshots()
  } catch (err) {
    fromSnapshot.todos = previousTodos
    publishRows()
    notifyError(err, 'Could not move the task')
  }
}

/** Hide one session's row from the overview panel (display-only, persisted —
 *  see `dismissed` above). Idempotent; republishes immediately so any
 *  subscriber of `$sessionTodoOverviewRows` repaints without a manual tick. */
export function dismissSessionTodoOverviewRow(storedSessionId: string): void {
  dismissed.add(storedSessionId)
  snapshots.delete(storedSessionId)
  dismissedItems.delete(storedSessionId)
  persistDismissed()
  persistSnapshots()
  persistDismissedItems()
  publishRows()
}

/** Evict a HARD-DELETED session's row entirely — unlike `dismissSessionTodoOverviewRow`,
 *  this does NOT add to `dismissed`. `dismissed` exists so a row can be un-hidden by
 *  fresh activity in that session (captureSnapshot un-dismisses on a new active list);
 *  a deleted session can never produce that activity again, so adding it there would
 *  only grow an ever-larger list of ids nothing will ever check again. Safe to call
 *  for an id that was never in the overview. */
export function removeSessionTodoOverviewRow(storedSessionId: string): void {
  const hadSnapshot = snapshots.delete(storedSessionId)
  const hadDismissedItems = dismissedItems.delete(storedSessionId)

  if (!hadSnapshot && !hadDismissedItems) {
    return
  }

  persistSnapshots()
  persistDismissedItems()
  publishRows()
}

/** Test-only reset — mirrors the pattern other session-scoped stores use.
 *  Clears BOTH the in-memory maps and their persisted copies. */
export function resetSessionTodoOverview(): void {
  snapshots.clear()
  dismissed.clear()
  dismissedItems.clear()
  persistSnapshots()
  persistDismissed()
  persistDismissedItems()
  publishRows()
}

// ── Startup backfill ────────────────────────────────────────────────────────
// `captureSnapshot` above only ever sees a session once THIS window observes
// it live (a todo.updated event, or a session.resume/activate response) — a
// session created in a different window, or before this app was last opened,
// never enters `snapshots` that way. `todo.list_open_sessions`
// (tui_gateway/methods_todo.py) answers "which stored sessions still have an
// unfinished todo list?" directly from the DB in one query, so a session like
// one left mid-checklist yesterday shows up here the first time the app is
// opened today, not only after the user happens to reopen that exact chat.
// Runs once per gateway connection (not on every reconnect-retry churn) and
// never overwrites a row `captureSnapshot` already populated — live data,
// however stale-looking, is always more authoritative than this cold read.
let backfillRequested = false

async function backfillFromStorage(): Promise<void> {
  const gateway = $gateway.get()

  if (!gateway) {
    return
  }

  try {
    const result = await ambientRequestFor(gateway)<{ sessions?: unknown }>('todo.list_open_sessions', {
      limit: 200
    })

    const rows = Array.isArray(result?.sessions) ? result.sessions : []
    let changed = false

    for (const row of rows) {
      if (!row || typeof row !== 'object') {
        continue
      }

      const storedSessionId = String((row as { session_id?: unknown }).session_id ?? '').trim()

      if (!storedSessionId || snapshots.has(storedSessionId) || dismissed.has(storedSessionId)) {
        continue
      }

      // This row came straight from the launch profile's DB (todo.list_open_sessions
      // has no session.list-style row to carry connection_id/profile), so the
      // owner ladder in session-states.ts (tile route → hint → row → runtime
      // ledger) has nothing to resolve UNLESS a hint already exists — a session
      // this window has never listed/opened otherwise 4001s the "Session owner
      // could not be resolved" fail-closed error the instant the user tries to
      // cancel an item (#todo-overview-backfill-owner). The backfill RPC only
      // ever reads the ambient connection's own profile DB, so "this window's
      // active connection/profile" is a true, not-a-guess owner for exactly
      // the rows it returns.
      if (!getSessionOwnerHint(storedSessionId)) {
        const connectionId = activeGatewayConnectionId()

        if (connectionId) {
          setSessionOwnerHint(storedSessionId, { connectionId, profile: activeGatewayProfileKey() })
        }
      }

      const todos = parseTodos({ todos: (row as { todos?: unknown }).todos })

      if (!todos || todos.length === 0) {
        continue
      }

      snapshots.set(storedSessionId, { todos, updatedAt: 0 })
      changed = true
    }

    if (changed) {
      persistSnapshots()
      publishRows()
    }
  } catch (err) {
    notifyError(err, 'Could not load existing task lists')
  }
}

// A plain `{ get, request }` gateway stub (as several tests supply) has no
// `.subscribe` — guard so those suites don't crash importing this module;
// production's real nanostores atom always has it.
if (typeof $gateway.subscribe === 'function') {
  $gateway.subscribe(gateway => {
    if (gateway && !backfillRequested) {
      backfillRequested = true
      void backfillFromStorage()
    }
  })
} else if ($gateway.get()) {
  backfillRequested = true
  void backfillFromStorage()
}
