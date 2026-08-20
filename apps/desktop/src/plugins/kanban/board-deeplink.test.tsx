// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { KanbanBoardPage } from './board'

// Network reads are stubbed; atoms ($boardSlug, $collapsedLanes, …) and query
// keys stay real so the page's store wiring is exercised, not replaced.
const fetchBoard = vi.fn()
const fetchTask = vi.fn()
const fetchLog = vi.fn()
const fetchBoards = vi.fn()
const fetchProfiles = vi.fn()
const fetchProjects = vi.fn()
const fetchOrchestration = vi.fn()

vi.mock('./api', async importOriginal => {
  const real = await importOriginal<typeof KanbanApi>()

  return {
    ...real,
    fetchBoard: (...args: unknown[]) => fetchBoard(...args),
    fetchTask: (...args: unknown[]) => fetchTask(...args),
    fetchLog: (...args: unknown[]) => fetchLog(...args),
    fetchBoards: (...args: unknown[]) => fetchBoards(...args),
    fetchProfiles: (...args: unknown[]) => fetchProfiles(...args),
    fetchProjects: (...args: unknown[]) => fetchProjects(...args),
    fetchOrchestration: (...args: unknown[]) => fetchOrchestration(...args)
  }
})

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/hermes', () => ({
  setApiRequestProfile: vi.fn()
}))

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

// A fresh board object per resolve — identity changes are exactly what the
// consumed-once guard must survive (refetch/poll after the user closed the
// drawer must not re-open it).
function boardWith(taskId: null | string) {
  return {
    columns: [
      { name: 'triage', tasks: [] },
      {
        name: 'todo',
        tasks: taskId ? [{ id: taskId, title: 'Deep link card', status: 'todo' }] : []
      },
      { name: 'ready', tasks: [] },
      { name: 'running', tasks: [] },
      { name: 'review', tasks: [] },
      { name: 'done', tasks: [] },
      { name: 'scheduled', tasks: [] }
    ],
    tenants: [],
    assignees: [],
    latest_event_id: 1,
    now: 0
  }
}

function detailFor(id: string) {
  return {
    task: { id, title: 'Deep link card', status: 'todo' },
    comments: [],
    events: [],
    attachments: [],
    links: { parents: [], children: [] },
    runs: []
  }
}

let queryClient: QueryClient

beforeEach(() => {
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  fetchBoard.mockResolvedValue(boardWith(null))
  fetchBoards.mockResolvedValue({ boards: [], current: 'default' })
  fetchProfiles.mockResolvedValue({ profiles: [] })
  fetchProjects.mockResolvedValue({ projects: [] })
  fetchOrchestration.mockResolvedValue({ default_assignee: '' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPage(entry: string) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[entry]}>
        <KanbanBoardPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('KanbanBoardPage deep link (?task=<id>)', () => {
  it('opens the drawer for a task id that lives on the board', async () => {
    fetchBoard.mockResolvedValue(boardWith('t_abc123'))
    fetchTask.mockResolvedValue(detailFor('t_abc123'))
    fetchLog.mockResolvedValue({ entries: [] })

    renderPage('/kanban?task=t_abc123')

    await waitFor(() => expect(fetchTask).toHaveBeenCalledWith('t_abc123'))
  })

  it('does not open a drawer for an id missing from the board, and still consumes the param', async () => {
    fetchBoard.mockResolvedValue(boardWith(null))

    renderPage('/kanban?task=t_ghost')

    // Let the board query land and the effect settle before asserting absence.
    await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    expect(fetchTask).not.toHaveBeenCalled()

    // A refetch after the foreign id was processed must stay silent — the
    // param was dropped even though nothing opened, so it can't leak onto a
    // later board where the id happens to exist.
    await act(async () => {
      queryClient.invalidateQueries({ queryKey: ['kanban', 'board'] })
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    expect(fetchTask).not.toHaveBeenCalled()
  })

  it('spends the param when the board fails to load, so a later board cannot be hit by it', async () => {
    // First load blows up — the board never resolves, but the link must still
    // be consumed. A stale ?task= left in the URL would fire against whatever
    // board loads next in this same mounted page.
    fetchBoard.mockRejectedValueOnce(new Error('board offline'))

    renderPage('/kanban?task=t_abc123')

    await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    expect(fetchTask).not.toHaveBeenCalled()

    // The board comes back (manual retry / board switch) — now containing the
    // id. The consumed param must not resurrect the deep link: no drawer, no
    // fetch, despite the id existing on the newly loaded board.
    fetchBoard.mockResolvedValue(boardWith('t_abc123'))
    await act(async () => {
      queryClient.invalidateQueries({ queryKey: ['kanban', 'board'] })
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    expect(fetchTask).not.toHaveBeenCalled()
  })

  it('does not re-open a consumed deep link when the board refetches', async () => {
    fetchBoard.mockResolvedValue(boardWith('t_abc123'))
    fetchTask.mockResolvedValue(detailFor('t_abc123'))
    fetchLog.mockResolvedValue({ entries: [] })

    renderPage('/kanban?task=t_abc123')

    await waitFor(() => expect(fetchTask).toHaveBeenCalledTimes(1))

    // A poll/socket refetch swaps the board object; the drawer must stay as
    // the user left it, not snap back open (and re-fetch) on every refetch.
    await act(async () => {
      queryClient.invalidateQueries({ queryKey: ['kanban', 'board'] })
      await new Promise(resolve => setTimeout(resolve, 50))
    })

    expect(fetchTask).toHaveBeenCalledTimes(1)
  })

  it('ignores the param on a plain /kanban route', async () => {
    fetchBoard.mockResolvedValue(boardWith(null))

    renderPage('/kanban')

    await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    expect(fetchTask).not.toHaveBeenCalled()
  })

  it('opens the drawer when the route gains the param after mount (in-app navigation)', async () => {
    fetchBoard.mockResolvedValue(boardWith('t_abc123'))
    fetchTask.mockResolvedValue(detailFor('t_abc123'))
    fetchLog.mockResolvedValue({ entries: [] })

    function NavigateToDeepLink() {
      const navigate = useNavigate()

      return (
        <>
          <KanbanBoardPage />
          <button onClick={() => navigate('/kanban?task=t_abc123')}>nav</button>
        </>
      )
    }

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/kanban']}>
          <NavigateToDeepLink />
        </MemoryRouter>
      </QueryClientProvider>
    )

    // Plain mount: no deep link yet.
    await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    expect(fetchTask).not.toHaveBeenCalled()

    // Navigating to the same route with the param is a deep link too.
    fireEvent.click(screen.getByRole('button', { name: 'nav' }))
    await waitFor(() => expect(fetchTask).toHaveBeenCalledWith('t_abc123'))
  })
})
