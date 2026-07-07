// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $kanbanActiveBoard,
  $kanbanBoard,
  $kanbanBoardError,
  $kanbanBoards,
  $kanbanRefreshing,
  $kanbanSelectedTaskId,
  KANBAN_COLUMNS
} from '@/store/kanban'
import type { KanbanBoardPayload, KanbanCard } from '@/types/kanban'

import { KanbanView } from './index'

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

const boardPayload: KanbanBoardPayload = {
  columns: KANBAN_COLUMNS.map(name => ({
    name,
    tasks:
      name === 'todo'
        ? [
            card('t_alpha', 'todo', {
              comment_count: 2,
              progress: { done: 1, total: 3 },
              warnings: { count: 1, kinds: { worker_crashed: 1 }, latest_at: 5, highest_severity: 'error' }
            })
          ]
        : name === 'running'
          ? [card('t_beta', 'running', { assignee: 'researcher', priority: 2 })]
          : []
  })),
  tenants: [],
  assignees: ['coder', 'researcher'],
  latest_event_id: 3,
  now: 1000
}

describe('KanbanView', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn(async (request: { path: string }) => {
      if (request.path.startsWith('/api/plugins/kanban/boards')) {
        return {
          boards: [{ slug: 'default', name: 'Default', description: '', icon: '', color: '', is_current: true }],
          current: 'default'
        }
      }

      if (request.path.startsWith('/api/plugins/kanban/board')) {
        return boardPayload
      }

      if (request.path.startsWith('/api/plugins/kanban/assignees')) {
        return { assignees: [] }
      }

      throw new Error(`unexpected path ${request.path}`)
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    window.localStorage.clear()
    $kanbanBoards.set([])
    $kanbanActiveBoard.set(null)
    $kanbanBoard.set(null)
    $kanbanBoardError.set(null)
    $kanbanRefreshing.set(false)
    $kanbanSelectedTaskId.set(null)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('renders every column and the mocked cards', async () => {
    render(<KanbanView onClose={() => undefined} />)

    // Cards from the mocked board payload.
    expect(await screen.findByText('Task t_alpha')).toBeTruthy()
    expect(screen.getByText('Task t_beta')).toBeTruthy()

    // All eight columns, in BOARD_COLUMNS order.
    for (const label of ['Triage', 'Todo', 'Scheduled', 'Ready', 'In progress', 'Blocked', 'Review', 'Done']) {
      expect(screen.getByText(label)).toBeTruthy()
    }

    // Card rollup chips: comments, progress, diagnostics badge.
    expect(screen.getByText('1/3')).toBeTruthy()

    // Header actions.
    expect(screen.getByText('Nudge dispatcher')).toBeTruthy()
    expect(screen.getByText('New card')).toBeTruthy()

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/plugins/kanban/board' }))
  })

  it('opens the card drawer when a card is selected', async () => {
    api.mockImplementation(async (request: { path: string }) => {
      if (request.path.startsWith('/api/plugins/kanban/boards')) {
        return { boards: [], current: 'default' }
      }

      if (request.path.startsWith('/api/plugins/kanban/board')) {
        return boardPayload
      }

      if (request.path.startsWith('/api/plugins/kanban/assignees')) {
        return { assignees: [{ name: 'coder', on_disk: true, counts: {} }] }
      }

      if (request.path.startsWith('/api/plugins/kanban/tasks/t_alpha')) {
        return {
          task: card('t_alpha', 'todo', { body: 'Do the thing', latest_summary: null }),
          comments: [{ id: 1, task_id: 't_alpha', author: 'coder', body: 'On it', created_at: 10 }],
          events: [{ id: 1, task_id: 't_alpha', kind: 'created', payload: null, created_at: 9, run_id: null }],
          attachments: [],
          links: { parents: [], children: [] },
          runs: []
        }
      }

      throw new Error(`unexpected path ${request.path}`)
    })

    $kanbanSelectedTaskId.set('t_alpha')
    render(<KanbanView onClose={() => undefined} />)

    expect(await screen.findByText('Do the thing')).toBeTruthy()
    expect(screen.getByText('On it')).toBeTruthy()
    expect(screen.getByText('Run history')).toBeTruthy()
  })
})
