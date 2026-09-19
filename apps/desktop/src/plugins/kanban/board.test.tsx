import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { KanbanBoardPage } from './board'
import { en, KANBAN_LOCALES } from './i18n'

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const board = {
  assignees: ['Ada'],
  columns: [
    {
      name: 'todo',
      tasks: [
        { assignee: null, id: 'null-assignee', status: 'todo', title: 'Null assignee' },
        { assignee: '', id: 'empty-assignee', status: 'todo', title: 'Empty assignee' },
        { assignee: 'Ada', id: 'assigned', status: 'todo', title: 'Assigned task' }
      ]
    }
  ],
  latest_event_id: 0,
  now: 0,
  tenants: []
}

const rest = vi.fn(async (path: string, _options?: PluginRestOptions): Promise<unknown> => {
  if (path === '/board') {
    return board
  }

  if (path === '/orchestration') {
    return { default_assignee: '', resolved_default_assignee: '' }
  }

  if (path === '/profiles') {
    return { profiles: [] }
  }

  throw new Error(`Unexpected REST request: ${path}`)
})

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, remove: vi.fn(), set: vi.fn() },
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

describe('board assignee filter', () => {
  it('filters both null and empty assignees through the localized Unassigned option', async () => {
    render(
      <QueryClientProvider client={client}>
        <KanbanBoardPage />
      </QueryClientProvider>
    )

    expect(await screen.findByText('Assigned task')).toBeTruthy()
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Filters' }), { button: 0, ctrlKey: false })
    fireEvent.click(await screen.findByRole('menuitem', { name: en.unassigned }))

    expect(await screen.findByText('Null assignee')).toBeTruthy()
    expect(screen.getByText('Empty assignee')).toBeTruthy()
    expect(screen.queryByText('Assigned task')).toBeNull()

  })
})
