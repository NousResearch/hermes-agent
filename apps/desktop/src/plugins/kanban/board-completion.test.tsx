import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { KanbanBoardPage } from './board'
import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanBoard } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const board = (result?: string): KanbanBoard => ({
  columns: [
    { name: 'todo', tasks: [{ id: 't_a', title: 'Approve the licence', status: 'todo', result }] },
    { name: 'ready', tasks: [] },
    { name: 'done', tasks: [] }
  ],
  tenants: [],
  assignees: [],
  latest_event_id: 0,
  now: 0
})

let current: KanbanBoard
let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const rest = vi.fn(async (path: string, _options?: PluginRestOptions): Promise<unknown> => {
  if (path === '/board' || path.startsWith('/board?')) {
    return current
  }

  if (path.startsWith('/boards')) {
    return { boards: [], current: 'default' }
  }

  if (path.startsWith('/profiles')) {
    return { profiles: [] }
  }

  if (path.startsWith('/orchestration')) {
    return { default_assignee: '' }
  }

  if (path.startsWith('/tasks/bulk')) {
    return { results: [{ id: 't_a', ok: true }] }
  }

  return { ok: true }
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
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

const calls = (method: string, prefix: string) =>
  rest.mock.calls.filter(([path, options]) => path.startsWith(prefix) && options?.method === method)

const openBoard = async () => {
  render(
    <QueryClientProvider client={client}>
      <KanbanBoardPage />
    </QueryClientProvider>
  )

  return screen.findByText('Approve the licence')
}

const dropOnDone = () => fireEvent.drop(screen.getByText(en.col.done.label), { dataTransfer: { getData: () => 't_a' } })

const summaryField = () => screen.findByPlaceholderText(en.completePlaceholder)

describe('completing a card from the board', () => {
  it('asks for a summary when a card with no result is dragged to Done', async () => {
    current = board()
    await openBoard()
    dropOnDone()

    fireEvent.change(await summaryField(), { target: { value: 'licence checked' } })
    fireEvent.click(screen.getByRole('button', { name: en.complete }))

    await waitFor(() =>
      expect(calls('PATCH', '/tasks/t_a')[0]?.[1]?.body).toEqual({ status: 'done', summary: 'licence checked' })
    )
  })

  it('moves a card that already holds a result without asking', async () => {
    current = board('licence checked')
    await openBoard()
    dropOnDone()

    await waitFor(() => expect(calls('PATCH', '/tasks/t_a')[0]?.[1]?.body).toEqual({ status: 'done' }))
    expect(screen.queryByPlaceholderText(en.completePlaceholder)).toBeNull()
  })

  it('sends one summary with a bulk move to Done', async () => {
    current = board()
    fireEvent.click(await openBoard(), { ctrlKey: true })

    // Radix opens the menu from the trigger's keyboard path under jsdom.
    fireEvent.keyDown(await screen.findByRole('button', { name: en.moveToShort }), { key: 'Enter' })
    fireEvent.click(await screen.findByRole('menuitem', { name: en.col.done.label }))

    expect(await screen.findByRole('heading', { name: en.completeTitle(1) })).toBeTruthy()
    fireEvent.change(await summaryField(), { target: { value: 'swept by hand' } })
    fireEvent.click(screen.getByRole('button', { name: en.complete }))

    await waitFor(() =>
      expect(calls('POST', '/tasks/bulk')[0]?.[1]?.body).toEqual({
        ids: ['t_a'],
        status: 'done',
        summary: 'swept by hand'
      })
    )
  })
})
