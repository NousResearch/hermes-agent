// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { parseKanbanApiError } from '@/kanban-api'
import type { KanbanBoardPayload, KanbanCard } from '@/types/kanban'

import {
  $kanbanActiveBoard,
  $kanbanBoard,
  $kanbanBoardError,
  $kanbanRefreshing,
  $kanbanSelectedTaskId,
  allowedDropColumns,
  applyKanbanCardMove,
  findKanbanCard,
  isAllowedDrop,
  KANBAN_COLUMNS,
  KANBAN_DRAG_TARGETS,
  moveKanbanTask,
  reconcileKanbanBoard,
  refreshKanbanBoard,
  setKanbanActiveBoard
} from './kanban'

function card(id: string, status: string, extra: Partial<KanbanCard> = {}): KanbanCard {
  return {
    id,
    title: `Task ${id}`,
    body: null,
    assignee: 'coder',
    status,
    priority: 0,
    created_by: 'test',
    created_at: 1,
    started_at: null,
    completed_at: null,
    workspace_kind: 'scratch',
    workspace_path: null,
    claim_lock: null,
    claim_expires: null,
    tenant: null,
    ...extra
  }
}

function boardWith(cards: KanbanCard[], latestEventId = 10): KanbanBoardPayload {
  return {
    columns: KANBAN_COLUMNS.map(name => ({
      name,
      tasks: cards.filter(entry => entry.status === name)
    })),
    tenants: [],
    assignees: ['coder'],
    latest_event_id: latestEventId,
    now: 1000
  }
}

function columnTasks(board: KanbanBoardPayload | null, name: string): string[] {
  return board?.columns.find(column => column.name === name)?.tasks.map(task => task.id) ?? []
}

describe('kanban transition rules', () => {
  it('never offers running as a drag target (dispatcher-claim only)', () => {
    expect(KANBAN_DRAG_TARGETS).not.toContain('running')

    for (const status of KANBAN_COLUMNS) {
      expect(allowedDropColumns(status)).not.toContain('running')
    }
  })

  it("excludes the card's own column from its drop targets", () => {
    expect(allowedDropColumns('todo')).not.toContain('todo')
    expect(allowedDropColumns('todo')).toEqual(['triage', 'scheduled', 'ready', 'blocked', 'review', 'done'])
  })

  it('allows dragging OFF running (reclaim back to the queue)', () => {
    expect(isAllowedDrop('running', 'ready')).toBe(true)
    expect(isAllowedDrop('ready', 'running')).toBe(false)
    expect(isAllowedDrop('todo', 'todo')).toBe(false)
  })
})

describe('reconcileKanbanBoard (polling cursor)', () => {
  it('keeps the previous atom identity when latest_event_id is unchanged', () => {
    const prev = boardWith([card('t_1', 'todo')], 42)
    // Same cursor, different `now`/age churn — must be treated as no change.
    const next = { ...boardWith([card('t_1', 'todo')], 42), now: 2000 }

    expect(reconcileKanbanBoard(prev, next)).toBe(prev)
  })

  it('adopts the new payload when the cursor advanced', () => {
    const prev = boardWith([card('t_1', 'todo')], 42)
    const next = boardWith([card('t_1', 'ready')], 43)

    expect(reconcileKanbanBoard(prev, next)).toBe(next)
    expect(reconcileKanbanBoard(null, next)).toBe(next)
  })
})

describe('board card helpers', () => {
  it('finds cards across columns and moves them immutably', () => {
    const board = boardWith([card('t_1', 'todo'), card('t_2', 'ready')])

    expect(findKanbanCard(board, 't_2')?.status).toBe('ready')
    expect(findKanbanCard(board, 't_x')).toBeNull()

    const moved = applyKanbanCardMove(board, 't_1', 'review')

    expect(columnTasks(moved, 'todo')).toEqual([])
    expect(columnTasks(moved, 'review')).toEqual(['t_1'])
    expect(findKanbanCard(moved, 't_1')?.status).toBe('review')
    // Original untouched.
    expect(columnTasks(board, 'todo')).toEqual(['t_1'])
  })
})

describe('moveKanbanTask', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    window.localStorage.clear()
    $kanbanBoard.set(boardWith([card('t_1', 'todo'), card('t_2', 'running')]))
    $kanbanBoardError.set(null)
    $kanbanRefreshing.set(false)
    $kanbanSelectedTaskId.set(null)
    $kanbanActiveBoard.set(null)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('PATCHes the status and reconciles from the server response', async () => {
    api.mockImplementation(async (request: { path: string; method?: string }) => {
      if (request.method === 'PATCH') {
        return { task: card('t_1', 'ready') }
      }

      // Post-mutation authoritative refetch.
      return boardWith([card('t_1', 'ready'), card('t_2', 'running')], 11)
    })

    const result = await moveKanbanTask('t_1', 'ready')

    expect(result).toEqual({ ok: true, status: 'ready', ruleMoved: false })
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/plugins/kanban/tasks/t_1',
        method: 'PATCH',
        body: { status: 'ready' }
      })
    )
    expect(columnTasks($kanbanBoard.get(), 'ready')).toContain('t_1')
    expect(columnTasks($kanbanBoard.get(), 'todo')).not.toContain('t_1')
  })

  it("renders the server's column when board rules redirect the transition", async () => {
    // Production backends can bounce done → review (review-required auto-inject).
    api.mockImplementation(async (request: { method?: string }) => {
      if (request.method === 'PATCH') {
        return { task: card('t_1', 'review') }
      }

      return boardWith([card('t_1', 'review'), card('t_2', 'running')], 11)
    })

    const result = await moveKanbanTask('t_1', 'done')

    expect(result.ok).toBe(true)
    expect(result.ruleMoved).toBe(true)
    expect(result.status).toBe('review')
    expect(columnTasks($kanbanBoard.get(), 'review')).toContain('t_1')
    expect(columnTasks($kanbanBoard.get(), 'done')).not.toContain('t_1')
  })

  it('reverts the optimistic move and surfaces the 409 detail', async () => {
    const snapshot = $kanbanBoard.get()

    // Model the real renderer shape: fetchJson's `409: {json}` rejection is
    // re-wrapped by ipcRenderer.invoke before it reaches the store.
    api.mockRejectedValue(
      new Error(
        'Error invoking remote method \'hermes:api\': Error: 409: {"detail": "Cannot move to \'ready\': blocked by parent(s) not done — \'Parent\' (t_9, status=todo)"}'
      )
    )

    const result = await moveKanbanTask('t_1', 'ready')

    expect(result.ok).toBe(false)
    expect(result.error).toContain('blocked by parent(s)')
    expect($kanbanBoard.get()).toBe(snapshot)
    expect(columnTasks($kanbanBoard.get(), 'todo')).toContain('t_1')
  })

  it('refuses running as a target without calling the backend', async () => {
    const result = await moveKanbanTask('t_1', 'running')

    expect(result.ok).toBe(false)
    expect(api).not.toHaveBeenCalled()
  })

  it('no-ops when dropped back on its own column', async () => {
    const result = await moveKanbanTask('t_1', 'todo')

    expect(result).toEqual({ ok: true, status: 'todo' })
    expect(api).not.toHaveBeenCalled()
  })
})

describe('refreshKanbanBoard', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    window.localStorage.clear()
    $kanbanBoard.set(null)
    $kanbanBoardError.set(null)
    $kanbanRefreshing.set(false)
    $kanbanActiveBoard.set(null)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('keeps atom identity across polls with an unchanged cursor', async () => {
    api.mockResolvedValue(boardWith([card('t_1', 'todo')], 7))

    await refreshKanbanBoard()
    const first = $kanbanBoard.get()

    await refreshKanbanBoard()

    expect($kanbanBoard.get()).toBe(first)
    expect(api).toHaveBeenCalledTimes(2)
  })

  it('scopes the fetch to the active board slug', async () => {
    $kanbanActiveBoard.set('atm10-server')
    api.mockResolvedValue(boardWith([], 1))

    await refreshKanbanBoard()

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/plugins/kanban/board?board=atm10-server' }))
  })

  it('falls back to the default board when the persisted slug 404s', async () => {
    $kanbanActiveBoard.set('deleted-board')
    api.mockRejectedValue(
      new Error(
        'Error invoking remote method \'hermes:api\': Error: 404: {"detail": "board \'deleted-board\' does not exist"}'
      )
    )

    await refreshKanbanBoard()

    expect($kanbanActiveBoard.get()).toBeNull()
    expect($kanbanBoardError.get()).toContain('does not exist')
  })
})

describe('setKanbanActiveBoard', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $kanbanActiveBoard.set(null)
    $kanbanBoard.set(boardWith([card('t_1', 'todo')]))
    $kanbanSelectedTaskId.set('t_1')
  })

  it('persists the slug and resets board-scoped state', () => {
    setKanbanActiveBoard('side-project')

    expect($kanbanActiveBoard.get()).toBe('side-project')
    expect(window.localStorage.getItem('hermes.desktop.kanban.activeBoard')).toBe('side-project')
    expect($kanbanBoard.get()).toBeNull()
    expect($kanbanSelectedTaskId.get()).toBeNull()
  })
})

describe('parseKanbanApiError', () => {
  it('extracts status and FastAPI detail', () => {
    const parsed = parseKanbanApiError(
      new Error('409: {"detail": "status transition to \'done\' not valid from current state"}')
    )

    expect(parsed.status).toBe(409)
    expect(parsed.detail).toBe("status transition to 'done' not valid from current state")
  })

  it('unwraps the Electron IPC prefix before parsing', () => {
    // window.hermesDesktop.api errors cross ipcRenderer.invoke, which rewrites
    // the rejection to this wrapped form — the production shape of every
    // kanban API error in the renderer.
    const parsed = parseKanbanApiError(
      new Error(
        'Error invoking remote method \'hermes:api\': Error: 409: {"detail": "Cannot move to \'ready\': blocked by parent(s) not done — \'Parent\' (t1, status=todo)"}'
      )
    )

    expect(parsed.status).toBe(409)
    expect(parsed.detail).toBe("Cannot move to 'ready': blocked by parent(s) not done — 'Parent' (t1, status=todo)")
  })

  it('unwraps the IPC prefix for non-HTTP errors too', () => {
    expect(parseKanbanApiError(new Error("Error invoking remote method 'hermes:api': Error: socket hang up"))).toEqual(
      {
        status: null,
        detail: 'socket hang up'
      }
    )
  })

  it('passes through non-HTTP errors', () => {
    expect(parseKanbanApiError(new Error('socket hang up'))).toEqual({ status: null, detail: 'socket hang up' })
  })

  it('tolerates non-JSON bodies', () => {
    expect(parseKanbanApiError(new Error('500: Internal Server Error'))).toEqual({
      status: 500,
      detail: 'Internal Server Error'
    })
  })
})
