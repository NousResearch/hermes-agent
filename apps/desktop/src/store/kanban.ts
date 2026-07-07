import { atom } from 'nanostores'

import { getKanbanBoard, getKanbanBoards, parseKanbanApiError, updateKanbanTask } from '@/kanban-api'
import { persistString, storedString } from '@/lib/storage'
import type { KanbanBoardInfo, KanbanBoardPayload, KanbanCard, KanbanTask } from '@/types/kanban'

// Column order mirrors BOARD_COLUMNS in plugins/kanban/dashboard/plugin_api.py.
export const KANBAN_COLUMNS = ['triage', 'todo', 'scheduled', 'ready', 'running', 'blocked', 'review', 'done'] as const

export type KanbanColumnName = (typeof KANBAN_COLUMNS)[number]

// 'running' is NEVER a drag target: ready → running happens only via the
// dispatcher's claim path, and the backend rejects a direct PATCH to
// 'running' with 400 (409 is reserved for invalid transitions / blocking
// parents). Everything
// else is a legal drop column (the server still owns transition semantics —
// review-required rules may bounce done → review, blocked stickiness may keep
// a card put; we render whatever it returns).
export const KANBAN_DRAG_TARGETS: readonly KanbanColumnName[] = KANBAN_COLUMNS.filter(name => name !== 'running')

/** Columns a card in `fromStatus` may be dropped on (excludes its own column). */
export function allowedDropColumns(fromStatus: string): KanbanColumnName[] {
  return KANBAN_DRAG_TARGETS.filter(name => name !== fromStatus)
}

export function isAllowedDrop(fromStatus: string, toColumn: string): boolean {
  return (allowedDropColumns(fromStatus) as string[]).includes(toColumn)
}

const BOARD_STORAGE_KEY = 'hermes.desktop.kanban.activeBoard'

export const $kanbanBoards = atom<KanbanBoardInfo[]>([])

// Active board slug. null → the backend's current board (env → current file →
// default). Client-side selection only (persisted locally) so picking a board
// here never flips the CLI/gateway's shared current-board pointer.
export const $kanbanActiveBoard = atom<null | string>(storedString(BOARD_STORAGE_KEY))

export const $kanbanBoard = atom<KanbanBoardPayload | null>(null)
export const $kanbanBoardError = atom<null | string>(null)
export const $kanbanSelectedTaskId = atom<null | string>(null)

// Polling state: true while a background refresh is in flight (prevents
// overlapping fetches when the interval outpaces a slow backend).
export const $kanbanRefreshing = atom(false)

export function setKanbanActiveBoard(slug: null | string): void {
  const next = slug || null

  if (next === $kanbanActiveBoard.get()) {
    return
  }

  persistString(BOARD_STORAGE_KEY, next)
  $kanbanActiveBoard.set(next)
  // Different board → old columns/selection are meaningless.
  $kanbanBoard.set(null)
  $kanbanBoardError.set(null)
  $kanbanSelectedTaskId.set(null)
}

export function setKanbanSelectedTaskId(id: null | string): void {
  $kanbanSelectedTaskId.set(id)
}

// latest_event_id is the cheap change cursor: an unchanged cursor means no
// task_events row was appended, so keep the previous atom value (identity)
// and skip all downstream re-render work. `now`/age fields churn every fetch,
// which is exactly why the cursor comparison replaces deep-equality here.
export function reconcileKanbanBoard(prev: KanbanBoardPayload | null, next: KanbanBoardPayload): KanbanBoardPayload {
  if (prev && prev.latest_event_id === next.latest_event_id) {
    return prev
  }

  return next
}

export async function refreshKanbanBoards(): Promise<void> {
  try {
    const { boards } = await getKanbanBoards()

    $kanbanBoards.set(boards)

    // A persisted slug whose board has since been deleted/archived would 404
    // every poll — fall back to the server's current board.
    const active = $kanbanActiveBoard.get()

    if (active && !boards.some(board => board.slug === active)) {
      setKanbanActiveBoard(null)
    }
  } catch {
    // Board list is auxiliary chrome; the board fetch surfaces real errors.
  }
}

export async function refreshKanbanBoard(): Promise<void> {
  if ($kanbanRefreshing.get()) {
    return
  }

  $kanbanRefreshing.set(true)

  try {
    const next = await getKanbanBoard($kanbanActiveBoard.get())

    $kanbanBoard.set(reconcileKanbanBoard($kanbanBoard.get(), next))
    $kanbanBoardError.set(null)
  } catch (error) {
    const parsed = parseKanbanApiError(error)

    if (parsed.status === 404 && $kanbanActiveBoard.get()) {
      // Stale persisted board — reset and let the next poll load the default.
      setKanbanActiveBoard(null)
    }

    $kanbanBoardError.set(parsed.detail)
  } finally {
    $kanbanRefreshing.set(false)
  }
}

export function findKanbanCard(board: KanbanBoardPayload | null, taskId: string): KanbanCard | null {
  if (!board) {
    return null
  }

  for (const column of board.columns) {
    const card = column.tasks.find(task => task.id === taskId)

    if (card) {
      return card
    }
  }

  return null
}

/** Immutably move a card to `toStatus`'s column (appended; server ordering is
 *  restored by the authoritative refetch). `patch` overlays server task fields. */
export function applyKanbanCardMove(
  board: KanbanBoardPayload,
  taskId: string,
  toStatus: string,
  patch?: Partial<KanbanTask>
): KanbanBoardPayload {
  const card = findKanbanCard(board, taskId)

  if (!card) {
    return board
  }

  const moved: KanbanCard = { ...card, ...patch, status: toStatus }

  return {
    ...board,
    columns: board.columns.map(column => {
      const without = column.tasks.filter(task => task.id !== taskId)
      const tasks = column.name === toStatus ? [...without, moved] : without

      return tasks === column.tasks ? column : { ...column, tasks }
    })
  }
}

export interface KanbanMoveResult {
  ok: boolean
  /** Status the server actually landed the card on (success only). */
  status?: string
  /** True when board rules redirected the move (e.g. done bounced to review). */
  ruleMoved?: boolean
  error?: string
}

// Server-authoritative move: optimistic column swap for immediate feedback,
// then reconcile from the PATCH response (production backends carry local
// transition patches — review-required auto-inject, blocked-initial-status
// stickiness — so the card may land somewhere other than the drop target).
// Errors (409 with an actionable detail) revert to the pre-drag snapshot.
export async function moveKanbanTask(taskId: string, targetStatus: string): Promise<KanbanMoveResult> {
  const board = $kanbanBoard.get()
  const card = findKanbanCard(board, taskId)

  if (!board || !card) {
    return { ok: false, error: 'task not on board' }
  }

  if (card.status === targetStatus) {
    return { ok: true, status: card.status }
  }

  if (!isAllowedDrop(card.status, targetStatus)) {
    return { ok: false, error: `cannot move to ${targetStatus}` }
  }

  const snapshot = board

  $kanbanBoard.set(applyKanbanCardMove(board, taskId, targetStatus))

  try {
    const response = await updateKanbanTask(taskId, { status: targetStatus }, $kanbanActiveBoard.get())
    const serverTask = response.task
    const serverStatus = serverTask?.status ?? targetStatus
    const current = $kanbanBoard.get()

    if (current) {
      $kanbanBoard.set(applyKanbanCardMove(current, taskId, serverStatus, serverTask ?? undefined))
    }

    // Refetch so rollups (progress, ready-recompute side effects on other
    // cards) reflect the server, not our local guess.
    void refreshKanbanBoard()

    return { ok: true, status: serverStatus, ruleMoved: serverStatus !== targetStatus }
  } catch (error) {
    $kanbanBoard.set(snapshot)

    return { ok: false, error: parseKanbanApiError(error).detail }
  }
}

/** Overlay a server-returned task onto its board card in place (post-PATCH
 *  reconcile for non-move edits: assignee, priority, title, body). */
export function applyKanbanTaskPatch(task: KanbanTask): void {
  const board = $kanbanBoard.get()

  if (!board || !findKanbanCard(board, task.id)) {
    return
  }

  $kanbanBoard.set(applyKanbanCardMove(board, task.id, task.status, task))
}
