import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { syncCronFailureToKanban, syncTodoToKanbanTasks } from './kanban-sync'

// ---------------------------------------------------------------------------
// Mock the preload API
// ---------------------------------------------------------------------------

const mockAllTasks = vi.fn<() => ReturnType<Window['hermesDesktop']['kanban']['allTasks']>>()
const mockCreateTask = vi.fn<(...args: Parameters<Window['hermesDesktop']['kanban']['createTask']>) => ReturnType<Window['hermesDesktop']['kanban']['createTask']>>()
const mockUpdateTask = vi.fn<(...args: Parameters<Window['hermesDesktop']['kanban']['updateTask']>) => ReturnType<Window['hermesDesktop']['kanban']['updateTask']>>()

vi.stubGlobal('window', {
  hermesDesktop: {
    kanban: {
      allTasks: mockAllTasks,
      createTask: mockCreateTask,
      updateTask: mockUpdateTask,
      boards: vi.fn(async () => []),
      comments: vi.fn(async () => []),
      addComment: vi.fn(async () => ({})),
      deleteComment: vi.fn(async () => ({ ok: true }))
    }
  }
})

beforeEach(() => {
  vi.clearAllMocks()
})

// ---------------------------------------------------------------------------
// syncTodoToKanbanTasks
// ---------------------------------------------------------------------------

describe('syncTodoToKanbanTasks', () => {
  it('updates mirrored task status when todo status changes', async () => {
    mockAllTasks.mockResolvedValue([
      {
        id: 'kanban-1',
        boardId: 'default',
        title: 'Write tests',
        description: '',
        status: 'todo',
        priority: 'medium',
        assignee: '',
        createdBy: '',
        createdAt: 1,
        updatedAt: 1,
        archived: false,
        order: 0,
        externalTaskId: 'todo-1',
        assigneeType: 'unassigned',
        sessionId: 'session-1',
        syncMode: 'linked'
      }
    ])
    mockUpdateTask.mockResolvedValue({} as any)

    await syncTodoToKanbanTasks('session-1', [
      { id: 'todo-1', content: 'Write tests', status: 'in_progress' }
    ])

    expect(mockUpdateTask).toHaveBeenCalledTimes(1)
    expect(mockUpdateTask).toHaveBeenCalledWith('kanban-1', {
      status: 'running',
      lastSyncedAt: expect.any(Number)
    })
  })

  it('sets done status for completed todos', async () => {
    mockAllTasks.mockResolvedValue([
      {
        id: 'kanban-2',
        boardId: 'default',
        title: 'Deploy',
        description: '',
        status: 'running',
        priority: 'medium',
        assignee: '',
        createdBy: '',
        createdAt: 1,
        updatedAt: 1,
        archived: false,
        order: 0,
        externalTaskId: 'todo-2',
        assigneeType: 'unassigned',
        sessionId: 'session-1',
        syncMode: 'mirrored'
      }
    ])
    mockUpdateTask.mockResolvedValue({} as any)

    await syncTodoToKanbanTasks('session-1', [
      { id: 'todo-2', content: 'Deploy', status: 'completed' }
    ])

    expect(mockUpdateTask).toHaveBeenCalledWith('kanban-2', {
      status: 'done',
      lastSyncedAt: expect.any(Number)
    })
  })

  it('does not match tasks without matching externalTaskId', async () => {
    mockAllTasks.mockResolvedValue([
      {
        id: 'kanban-3',
        boardId: 'default',
        title: 'Random',
        description: '',
        status: 'todo',
        priority: 'medium',
        assignee: '',
        createdBy: '',
        createdAt: 1,
        updatedAt: 1,
        archived: false,
        order: 0,
        externalTaskId: undefined,
        assigneeType: 'unassigned',
        sessionId: 'session-1',
        syncMode: 'linked'
      }
    ])

    await syncTodoToKanbanTasks('session-1', [
      { id: 'todo-3', content: 'Something', status: 'in_progress' }
    ])

    expect(mockUpdateTask).not.toHaveBeenCalled()
  })

  it('does nothing when no tasks match the session', async () => {
    mockAllTasks.mockResolvedValue([])

    await syncTodoToKanbanTasks('session-other', [
      { id: 'todo-1', content: 'Task', status: 'in_progress' }
    ])

    expect(mockUpdateTask).not.toHaveBeenCalled()
  })
})

// ---------------------------------------------------------------------------
// syncCronFailureToKanban
// ---------------------------------------------------------------------------

describe('syncCronFailureToKanban', () => {
  it('creates blocked task for a new cron error', async () => {
    mockAllTasks.mockResolvedValue([])
    mockCreateTask.mockResolvedValue({} as any)

    await syncCronFailureToKanban([
      { id: 'cron-1', name: 'Nightly Backup', last_error: 'Connection timeout' }
    ])

    expect(mockCreateTask).toHaveBeenCalledTimes(1)
    expect(mockCreateTask).toHaveBeenCalledWith(
      expect.objectContaining({
        title: expect.stringContaining('Nightly Backup'),
        description: 'Connection timeout',
        status: 'blocked',
        priority: 'high',
        source: 'cron',
        externalTaskId: 'cron-1',
        externalTaskKind: 'cron_job'
      })
    )
  })

  it('does not create duplicate task for the same error', async () => {
    mockAllTasks.mockResolvedValue([])
    mockCreateTask.mockResolvedValue({} as any)

    // First call — creates
    await syncCronFailureToKanban([
      { id: 'cron-2', name: 'Sync', last_error: 'Disk full' }
    ])
    expect(mockCreateTask).toHaveBeenCalledTimes(1)

    // Second call with same error — should NOT create
    await syncCronFailureToKanban([
      { id: 'cron-2', name: 'Sync', last_error: 'Disk full' }
    ])
    expect(mockCreateTask).toHaveBeenCalledTimes(1)
  })

  it('does not create task if a blocked task already exists for the error', async () => {
    mockAllTasks.mockResolvedValue([
      {
        id: 'existing-blocked',
        boardId: 'default',
        title: 'Cron failed: Cleanup',
        description: 'OOM killed',
        status: 'blocked',
        priority: 'high',
        assignee: '',
        createdBy: '',
        createdAt: 1,
        updatedAt: 1,
        archived: false,
        order: 0,
        externalTaskId: 'cron-3',
        externalTaskKind: 'cron_job',
        assigneeType: 'unassigned',
        sessionId: undefined,
        syncMode: 'linked'
      }
    ])

    await syncCronFailureToKanban([
      { id: 'cron-3', name: 'Cleanup', last_error: 'OOM killed' }
    ])

    expect(mockCreateTask).not.toHaveBeenCalled()
  })

  it('clears tracking when job becomes healthy', async () => {
    mockAllTasks.mockResolvedValue([])
    mockCreateTask.mockResolvedValue({} as any)

    // Error
    await syncCronFailureToKanban([
      { id: 'cron-4', name: 'Report', last_error: 'Timeout' }
    ])
    expect(mockCreateTask).toHaveBeenCalledTimes(1)

    // Cleared (no error)
    mockCreateTask.mockClear()
    await syncCronFailureToKanban([
      { id: 'cron-4', name: 'Report', last_error: null }
    ])
    expect(mockCreateTask).not.toHaveBeenCalled()

    // New error — should create again (previous was cleared)
    await syncCronFailureToKanban([
      { id: 'cron-4', name: 'Report', last_error: 'DNS resolution failed' }
    ])
    expect(mockCreateTask).toHaveBeenCalledTimes(1)
  })
})
