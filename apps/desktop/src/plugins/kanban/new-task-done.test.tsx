import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Exercise the real board and New task dialog; only the REST boundary is mocked.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug } from './api'
import { KanbanBoardPage } from './board'
import { en, KANBAN_LOCALES } from './i18n'

const { createTaskMock, patchTaskMock } = vi.hoisted(() => ({
  createTaskMock: vi.fn(),
  patchTaskMock: vi.fn()
}))

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoard: vi.fn(async () => ({
    assignees: [],
    columns: [
      { name: 'ready', tasks: [] },
      { name: 'done', tasks: [] }
    ],
    latest_event_id: 0,
    now: 0,
    tenants: []
  })),
  fetchBoards: vi.fn(async () => ({
    boards: [{ name: 'Main', project_id: null, slug: 'main', total: 0 }],
    current: 'main'
  })),
  fetchOrchestration: vi.fn(async () => ({ resolved_default_assignee: 'default' })),
  fetchProfiles: vi.fn(async () => ({ profiles: [] })),
  createTask: createTaskMock,
  patchTask: patchTaskMock
}))

let disposeLocales: () => void

beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  createTaskMock.mockReset().mockResolvedValue({ task: { id: 't_manual', status: 'ready' } })
  patchTaskMock.mockReset().mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  disposeLocales()
  $boardSlug.set('')
  vi.clearAllMocks()
})

describe('new task in Done', () => {
  it('requires and persists a completion summary when creating directly in Done', async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <KanbanBoardPage />
      </QueryClientProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: en.newTaskIn(en.col.done.label) }))
    fireEvent.change(screen.getByPlaceholderText(en.titlePlaceholder), { target: { value: 'Manually finished' } })

    const create = screen.getByRole('button', { name: en.createTask }) as HTMLButtonElement
    expect(create.disabled).toBe(true)
    expect(createTaskMock).not.toHaveBeenCalled()

    fireEvent.change(screen.getByRole('textbox', { name: 'Completion summary' }), {
      target: { value: '  Verified in person  ' }
    })

    expect(create.disabled).toBe(false)
    fireEvent.click(create)

    await waitFor(() => {
      expect(createTaskMock).toHaveBeenCalledWith(expect.objectContaining({ title: 'Manually finished' }))
      expect(patchTaskMock).toHaveBeenCalledWith('t_manual', {
        result: 'Verified in person',
        status: 'done',
        summary: 'Verified in person'
      })
    })
  })
})
