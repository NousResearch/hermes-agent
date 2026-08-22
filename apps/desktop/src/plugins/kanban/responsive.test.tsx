import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { fetchBoard, fetchBoards, fetchLog, fetchTask } from './api'
import { KanbanBoardPage } from './board'
import { TaskDrawer } from './drawer'
import type { KanbanBoard, KanbanTaskDetail } from './types'

vi.mock('./api', async importOriginal => {
  const actual = await importOriginal<typeof KanbanApi>()

  return {
    ...actual,
    fetchBoard: vi.fn(),
    fetchBoards: vi.fn(),
    fetchLog: vi.fn(),
    fetchTask: vi.fn()
  }
})

const board: KanbanBoard = {
  assignees: ['canary-worker'],
  columns: [
    {
      name: 'running',
      tasks: [{ assignee: 'canary-worker', id: 'task-safe', status: 'running', title: 'Privacy-safe evidence task' }]
    },
    { name: 'review', tasks: [{ id: 'task-review', status: 'review', title: 'Production review task' }] }
  ],
  latest_event_id: 2,
  now: 1_800_000_000,
  tenants: []
}

const detail: KanbanTaskDetail = {
  attachments: [],
  comments: [],
  events: [],
  links: { children: [], parents: [] },
  runs: [],
  task: { assignee: 'canary-worker', id: 'task-safe', status: 'running', title: 'Privacy-safe evidence task' }
}

function resizeWindow(width: number) {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: width, writable: true })
  window.dispatchEvent(new Event('resize'))
}

function providers(child: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false }, queries: { retry: false } } })

  return <QueryClientProvider client={client}>{child}</QueryClientProvider>
}

beforeEach(() => {
  vi.mocked(fetchBoard).mockResolvedValue(board)
  vi.mocked(fetchBoards).mockResolvedValue({ boards: [{ default_workspace_kind: 'scratch', slug: 'default' }], current: 'default' })
  vi.mocked(fetchTask).mockResolvedValue(detail)
  vi.mocked(fetchLog).mockResolvedValue({ content: '', exists: false, size_bytes: 0, truncated: false })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('production Kanban responsive containment', () => {
  for (const width of [320, 360, 390, 430]) {
    it(`contains the production board toolbar and scrolls production lanes at ${width}px`, async () => {
      resizeWindow(width)
      const { container } = render(providers(<KanbanBoardPage />))

      await screen.findByText('Privacy-safe evidence task')
      const page = container.firstElementChild as HTMLElement
      const toolbar = page.querySelector<HTMLElement>('header[data-kanban-layout]')!
      const laneStrip = screen.getByText('Privacy-safe evidence task').closest('[style*="width"]')!.parentElement as HTMLElement
      const lanes = [...laneStrip.children] as HTMLElement[]

      expect(page.className).toContain('overflow-hidden')
      expect(toolbar.dataset.kanbanLayout).toBe('mobile')
      expect(toolbar.className).toContain('grid-cols-[minmax(0,1fr)_auto]')
      expect(toolbar.querySelector('input')).toBeTruthy()
      expect(laneStrip.className).toContain('overflow-x-auto')
      expect(laneStrip.className).toContain('snap-mandatory')
      expect(lanes).toHaveLength(2)
      expect(lanes.every(lane => lane.style.width === `${width - 32}px`)).toBe(true)
      expect(lanes.every(lane => Number.parseFloat(lane.style.width) <= width)).toBe(true)
    })

    it(`contains the production task drawer at ${width}px`, async () => {
      resizeWindow(width)
      const { container } = render(providers(
        <div className="relative h-screen w-full overflow-hidden">
          <TaskDrawer columns={['running', 'review']} id="task-safe" onClose={() => undefined} onOpen={() => undefined} />
        </div>
      ))

      await waitFor(() => expect(fetchTask).toHaveBeenCalledWith('task-safe'))
      const drawer = container.querySelector<HTMLElement>('[data-kanban-layout]')!

      expect(drawer.dataset.kanbanLayout).toBe('mobile')
      expect(drawer.className).toContain('w-full')
      expect(drawer.className).toContain('max-w-full')
      expect(drawer.style.width).toBe(`${width}px`)
      expect(Number.parseFloat(drawer.style.width)).toBeLessThanOrEqual(width)
    })
  }

  it('preserves production desktop lanes and drawer geometry', async () => {
    resizeWindow(1024)
    const boardView = render(providers(<KanbanBoardPage />))
    await screen.findByText('Privacy-safe evidence task')
    const toolbar = boardView.container.querySelector<HTMLElement>('header[data-kanban-layout]')!
    const lane = screen.getByText('Privacy-safe evidence task').closest<HTMLElement>('[style*="width"]')!

    expect(toolbar.dataset.kanbanLayout).toBe('desktop')
    expect(lane.style.width).toBe('256px')
    boardView.unmount()

    const drawerView = render(providers(<TaskDrawer columns={['running']} id="task-safe" onClose={() => undefined} onOpen={() => undefined} />))
    await waitFor(() => expect(fetchTask).toHaveBeenCalled())
    expect(drawerView.container.querySelector<HTMLElement>('[data-kanban-layout]')?.style.width).toBe('416px')
  })
})
