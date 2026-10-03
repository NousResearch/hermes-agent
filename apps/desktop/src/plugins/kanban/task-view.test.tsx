import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness drives the host's request scope, as a connection switch does.
// eslint-disable-next-line no-restricted-imports
import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'
// eslint-disable-next-line no-restricted-imports
import { clearNotifications } from '@/store/notifications'

import { $taskView, bindApi } from './api'
import { TaskDrawer } from './drawer'
import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanTaskDetail } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const baseDetail: Omit<KanbanTaskDetail, 'attachments'> = {
  task: { id: 't_example', title: 'Example task', body: 'Keep this description readable.', status: 'todo' },
  comments: [{ id: 1, author: 'test', body: 'Keep this comment readable.', created_at: 0 }],
  events: [],
  links: { parents: [], children: [] },
  runs: []
}

let detail: object
let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const rest = vi.fn(async (path: string, options?: PluginRestOptions): Promise<unknown> => {
  if (path.startsWith('/tasks/t_example/comments') && options?.method === 'POST') {
    return { ok: true }
  }

  if (path === '/tasks/t_example') {
    return detail
  }

  if (path.startsWith('/tasks/t_example/log?')) {
    return { exists: false, content: '', size_bytes: 0, truncated: false }
  }

  if (path === '/profiles') {
    return { profiles: [] }
  }

  if (path === '/orchestration') {
    return { default_assignee: '' }
  }

  throw new Error(`Unexpected REST request: ${path}`)
})

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  clearNotifications()
  vi.unstubAllGlobals()
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

function openDrawer() {
  return render(
    <QueryClientProvider client={client}>
      <TaskDrawer columns={['todo', 'ready', 'done']} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
    </QueryClientProvider>
  )
}

// ISSUE #123662: the centered task modal replaced the board-side drawer with
// no way back, and commenting on a running task means scrolling past the whole
// detail. The task must offer a Drawer <-> Expanded switch, and a running
// task must pin its message box above the scroll content.
describe('task view switch (issue #123662)', () => {
  it('offers a drawer view toggle in the task header', async () => {
    detail = { ...baseDetail, attachments: [] }
    openDrawer()

    expect(await screen.findByRole('heading', { name: 'Example task' })).toBeTruthy()
    expect(await screen.findByRole('button', { name: 'Drawer view' })).toBeTruthy()
  })

  it('pins a message box above the detail for running tasks', async () => {
    detail = {
      ...baseDetail,
      attachments: [],
      task: { ...baseDetail.task, status: 'running', assignee: 'worker', worker_pid: 123 }
    }
    openDrawer()

    expect(await screen.findByRole('heading', { name: 'Example task' })).toBeTruthy()
    expect(await screen.findByTestId('task-quick-comment')).toBeTruthy()
    // The pinned box owns the composer while running: the feed must not stack
    // a second live copy under it.
    expect(screen.getAllByPlaceholderText(en.messageWorker)).toHaveLength(1)
  })

  it('switches to the board-side drawer and back without losing the task', async () => {
    detail = { ...baseDetail, attachments: [] }
    openDrawer()

    expect(await screen.findByRole('heading', { name: 'Example task' })).toBeTruthy()
    fireEvent.click(await screen.findByRole('button', { name: 'Drawer view' }))

    expect(await screen.findByRole('button', { name: 'Expanded view' })).toBeTruthy()
    expect(screen.getByRole('dialog').getAttribute('data-task-view')).toBe('drawer')
    expect(screen.getByRole('heading', { name: 'Example task' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Expanded view' }))
    expect(await screen.findByRole('button', { name: 'Drawer view' })).toBeTruthy()
    expect(screen.getByRole('dialog').getAttribute('data-task-view')).toBe('expanded')
  })

  it('preserves the selected feed tab when switching views', async () => {
    detail = {
      ...baseDetail,
      attachments: [],
      events: [{ id: 1, kind: 'created', payload: null, created_at: 0 }]
    }
    openDrawer()

    fireEvent.click(await screen.findByRole('button', { name: en.activity(1) }))
    expect(await screen.findByText('created')).toBeTruthy()

    fireEvent.click(await screen.findByRole('button', { name: 'Drawer view' }))
    expect(screen.getByRole('button', { name: en.activity(1), pressed: true })).toBeTruthy()
    expect(screen.getByText('created')).toBeTruthy()
  })

  it('remembers the last-used view when the task is reopened', async () => {
    detail = { ...baseDetail, attachments: [] }
    const first = openDrawer()

    fireEvent.click(await screen.findByRole('button', { name: 'Drawer view' }))
    expect($taskView.get()).toBe('drawer')
    first.unmount()

    openDrawer()
    expect(await screen.findByRole('button', { name: 'Expanded view' })).toBeTruthy()
    expect(screen.getByRole('dialog').getAttribute('data-task-view')).toBe('drawer')
  })
})
