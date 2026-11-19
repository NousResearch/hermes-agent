import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The harness uses the same contribution and locale boundaries as the host.
// eslint-disable-next-line no-restricted-imports
import { ContribRender } from '@/contrib/react/boundary'
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { KanbanBoardPage } from './board'
import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanBoard, KanbanTask } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const task: KanbanTask = { id: 't_close', title: 'Keep this task', status: 'todo' }

const board: KanbanBoard = {
  columns: [{ name: 'todo', tasks: [task] }],
  tenants: [],
  assignees: [],
  latest_event_id: 0,
  now: 0
}

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void
let boardResult: () => Promise<KanbanBoard>

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  boardResult = async () => board
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, _options?: PluginRestOptions): Promise<T> => {
      const route = path.split('?')[0]

      const data = {
        '/boards': { boards: [], current: '' },
        '/profiles': { profiles: [] },
        '/orchestration': { default_assignee: '' },
        '/tasks/t_close': { task, comments: [], events: [], links: { parents: [], children: [] }, runs: [] },
        '/tasks/t_close/log': { exists: false, content: '', size_bytes: 0, truncated: false }
      }

      if (route === '/board') {
        return (await boardResult()) as T
      }

      if (!(route in data)) {
        throw new Error(`Unexpected REST request: ${path}`)
      }

      return data[route as keyof typeof data] as T
    },
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.restoreAllMocks()
})

function openBoard() {
  const close = vi.fn()

  const result = render(
    <QueryClientProvider client={client}>
      <ContribRender onClose={close} render={KanbanBoardPage} />
    </QueryClientProvider>
  )

  return { ...result, close }
}

describe('Kanban close controls', () => {
  it('uses the host close action for both the header button and Escape', async () => {
    const { close } = openBoard()
    await screen.findByText(task.title)
    const button = screen.getByRole('button', { name: en.close })
    fireEvent.click(button)
    expect(close).toHaveBeenCalledTimes(1)
    fireEvent.keyDown(button, { key: 'Escape' })
    expect(close).toHaveBeenCalledTimes(2)
  })

  it('dismisses the drawer, then the selection, then the board', async () => {
    const { close } = openBoard()
    const card = await screen.findByText(task.title)
    fireEvent.click(card, { ctrlKey: true })
    expect(screen.getByRole('button', { name: en.clearSelection })).toBeTruthy()
    fireEvent.click(card)
    await screen.findByRole('heading', { name: task.title })

    fireEvent.keyDown(card, { key: 'Escape' })
    expect(screen.queryByRole('heading', { name: task.title })).toBeNull()
    expect(screen.getByRole('button', { name: en.clearSelection })).toBeTruthy()
    expect(close).not.toHaveBeenCalled()

    fireEvent.keyDown(card, { key: 'Escape' })
    expect(screen.queryByRole('button', { name: en.clearSelection })).toBeNull()
    expect(close).not.toHaveBeenCalled()
    fireEvent.keyDown(card, { key: 'Escape' })
    expect(close).toHaveBeenCalledOnce()
  })

  it('leaves search, composition, and keys outside the board alone', async () => {
    const { close } = openBoard()
    await screen.findByText(task.title)
    fireEvent.keyDown(screen.getByRole('textbox', { name: en.filterCards }), { key: 'Escape' })
    fireEvent.keyDown(screen.getByRole('button', { name: en.close }), {
      key: 'Escape',
      isComposing: true
    })
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(close).not.toHaveBeenCalled()
  })

  it('lets a nested dialog consume Escape without closing the board', async () => {
    const { close } = openBoard()
    await screen.findByText(task.title)
    fireEvent.click(screen.getByRole('button', { name: en.newTask }))
    const dialog = await screen.findByRole('dialog')
    fireEvent.keyDown(dialog, { key: 'Escape' })
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(close).not.toHaveBeenCalled()
  })

  it.each(['loading', 'empty', 'error'] as const)('keeps Close available while %s', async state => {
    boardResult = () =>
      state === 'loading'
        ? new Promise(() => {})
        : state === 'error'
          ? Promise.reject(new Error('Board unavailable'))
          : Promise.resolve({ ...board, columns: [] })
    const { close } = openBoard()

    if (state === 'error') {
      await screen.findByText('Board unavailable')
    }

    fireEvent.click(screen.getByRole('button', { name: en.close }))
    expect(close).toHaveBeenCalledOnce()
  })
})
