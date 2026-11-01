import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The Kanban Sessions view remains a separate read-only surface. Its only
// conversion path asks for user-authored task fields; it never copies messages.
vi.mock('./api', async importOriginal => ({
  ...(await importOriginal()),
  fetchSessionMirrors: vi.fn(async () => ({ mirrors: [{
    id: 12,
    profile: 'worker',
    platform: 'telegram',
    chat_id: 'chat-1',
    thread_id: 'topic-3',
    session_id: 'session-12',
    title: 'telegram session',
    status: 'completed',
    received_at: 1_700_000_000,
    started_at: 1_700_000_010,
    completed_at: 1_700_000_020,
    updated_at: 1_700_000_020,
    archived_at: null,
    promoted_task_id: null,
  }] })),
  archiveSessionMirror: vi.fn(async () => ({ ok: true })),
  deleteSessionMirror: vi.fn(async () => ({ deleted: true })),
  promoteSessionMirror: vi.fn(async () => ({ task_id: 'task-12' })),
}))

// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { archiveSessionMirror, deleteSessionMirror, promoteSessionMirror } from './api'
import { KANBAN_LOCALES } from './i18n'
import { KanbanSessionsView } from './sessions'

let disposeLocales: () => void = () => undefined

const mount = () => render(
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <KanbanSessionsView />
  </QueryClientProvider>,
)

afterEach(() => {
  cleanup()
  disposeLocales()
  vi.clearAllMocks()
})

beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

describe('Kanban session mirrors', () => {
  it('renders metadata and lifecycle times as a separate non-draggable read-only record', async () => {
    mount()

    expect(await screen.findByRole('region', { name: 'Sessions' })).toBeTruthy()
    expect(await screen.findByText('telegram session')).toBeTruthy()
    expect(screen.getByText('Read-only session mirrors — not tasks and never dispatched.')).toBeTruthy()
    const item = screen.getByRole('listitem')
    expect(within(item).getByText((_, element) =>
      element?.tagName === 'SPAN' && element.textContent?.replace(/\s+/g, ' ').includes('telegram · worker · chat-1') === true,
    )).toBeTruthy()
    expect(within(item).getByText((_, element) =>
      element?.tagName === 'DIV' && element.textContent?.replace(/\s+/g, ' ').includes('Session session-12 · Thread topic-3') === true,
    )).toBeTruthy()

    for (const label of ['Received', 'Started', 'Completed']) {
      expect(within(item).getByText((_, element) =>
        element?.tagName === 'TIME' && element.textContent?.includes(label) === true,
      )).toBeTruthy()
    }

    expect(item.getAttribute('draggable')).toBeNull()
  })

  it('requires a user title for promotion and exposes only explicit archive/delete actions', async () => {
    mount()

    const promote = await screen.findByRole('button', { name: 'Promote to task' })
    expect(promote.hasAttribute('disabled')).toBe(true)
    fireEvent.change(screen.getByLabelText('Task title (required)'), { target: { value: 'Review the request' } })
    fireEvent.change(screen.getByLabelText('Task description (optional)'), { target: { value: 'User-authored description' } })
    expect(promote.hasAttribute('disabled')).toBe(false)
    fireEvent.click(promote)
    await waitFor(() => expect(promoteSessionMirror).toHaveBeenCalledWith(12, 'Review the request', 'User-authored description'))

    fireEvent.click(screen.getByRole('button', { name: 'Archive session' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete session' }))
    await waitFor(() => {
      expect(vi.mocked(archiveSessionMirror).mock.calls[0]?.[0]).toBe(12)
      expect(vi.mocked(deleteSessionMirror).mock.calls[0]?.[0]).toBe(12)
    })
  })
})
