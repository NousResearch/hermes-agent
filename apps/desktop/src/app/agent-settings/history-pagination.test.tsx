import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $activeGatewayProfile } from '@/store/profile'
import { makeSessionInfo } from '@/test/session-info'

import { AgentHistory } from './history'
import type * as HistoryData from './history-data'
import { deleteHistorySelection, mutateHistorySession, previewHistoryCleanup } from './history-data'
import { HistoryCleanupDialog } from './history-dialogs'

const { requests } = vi.hoisted(() => ({ requests: [] as { offset: number; filters: HistoryData.HistoryFilters }[] }))
vi.mock('../open-session', () => ({ openSession: vi.fn() }))
vi.mock('./history-data', async importOriginal => ({
  ...(await importOriginal<typeof HistoryData>()),
  getHistoryRunning: async () => new Set<string>(),
  previewHistoryCleanup: vi.fn(),
  deleteHistorySelection: vi.fn(),
  mutateHistorySession: vi.fn(),
  getHistoryPage: async (_owner: unknown, filters: HistoryData.HistoryFilters, offset = 0) => {
    requests.push({ filters, offset })

    if (filters.archived === 'only') {
      return {
        limit: 25,
        offset,
        total: 1,
        sessions: [
          makeSessionInfo({ id: 'archived', title: 'Archived conversation', archived: true, source: 'desktop' }),
          makeSessionInfo({ id: 'pin', title: 'Live pinned backfill', pinned: true, source: 'desktop' }),
          makeSessionInfo({ id: 'canonical', title: 'Bot Chat', hidden: true, archived: true })
        ]
      }
    }

    if (filters.source.length) {
      return {
        limit: 25,
        offset,
        total: 1,
        sessions: [
          makeSessionInfo({ id: 'schedule', title: 'Scheduled run', source: 'cron' }),
          makeSessionInfo({ id: 'discord', title: 'Discord thread', source: 'discord' }),
          makeSessionInfo({ id: 'pin', title: 'Live pinned backfill', pinned: true, source: 'desktop' })
        ]
      }
    }

    const sessions = Array.from({ length: Math.min(25, 501 - offset) }, (_, index) =>
      makeSessionInfo({
        id: `history-${offset + index}`,
        title: `History entry ${offset + index}`,
        profile: 'research',
        source: 'desktop'
      })
    )

    if (offset + 25 <= 499) {
      sessions.push(
        makeSessionInfo({
          id: 'history-499',
          title: 'History entry 499',
          profile: 'research',
          pinned: true,
          source: 'desktop'
        })
      )
    }

    return { limit: 25, offset, total: 501, sessions }
  }
}))

const clients: QueryClient[] = []
afterEach(() => {
  cleanup()
  clients.splice(0).forEach(client => client.clear())
  $apiRequestScope.set({ profile: null, connectionId: null })
  $activeGatewayProfile.set('default')
  requests.length = 0
  vi.mocked(previewHistoryCleanup).mockReset()
  vi.mocked(deleteHistorySelection).mockReset()
  vi.mocked(mutateHistorySession).mockReset()
})

function renderHistory() {
  $activeGatewayProfile.set('research')
  $apiRequestScope.set({ profile: 'research', connectionId: 'alpha' })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  clients.push(client)

  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <AgentHistory onClose={() => {}} owner={{ connectionId: 'alpha', profile: 'research' }} />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

it('keeps earlier history reachable beyond 500 rows and deduplicates pinned backfills', async () => {
  renderHistory()
  await screen.findByText('History entry 24')
  expect(screen.queryByText('All sessions loaded')).toBeNull()
  const loadMore = screen.getByRole('button', { name: 'Load more' })

  for (let offset = 25; offset <= 500; offset += 25) {
    fireEvent.click(loadMore)
    await screen.findByText(`History entry ${offset === 500 ? 500 : offset + 23}`)
  }

  expect(screen.getByText('History entry 0')).toBeTruthy()
  expect(screen.getAllByText('History entry 499')).toHaveLength(1)
  expect(loadMore.isConnected).toBe(false)
  expect(requests.map(request => request.offset)).toEqual(Array.from({ length: 21 }, (_, index) => index * 25))
}, 30_000)

it('searches the remaining pages instead of calling a loaded-subset miss empty history', async () => {
  renderHistory()
  await screen.findByText('History entry 24')
  fireEvent.change(screen.getByRole('textbox', { name: 'Search titles, previews, or IDs' }), {
    target: { value: 'History entry 500' }
  })
  await screen.findByText('History entry 500', {}, { timeout: 15_000 })
  expect(screen.queryByText('History entry 24')).toBeNull()
  expect(screen.getByText('All sessions loaded')).toBeTruthy()
}, 20_000)

it('applies source unions and archive boundaries to backfilled pins without exposing canonical chats', async () => {
  renderHistory()
  await screen.findByText('History entry 24')
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }))
  fireEvent.click(screen.getByRole('checkbox', { name: 'Schedule' }))
  await screen.findByText('Scheduled run')
  expect(screen.queryByText('Discord thread')).toBeNull()
  expect(screen.queryByText('Live pinned backfill')).toBeNull()
  fireEvent.click(screen.getByRole('checkbox', { name: 'Discord' }))
  await screen.findByText('Discord thread')
  fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
  fireEvent.click(screen.getByRole('radio', { name: 'Only' }))
  await screen.findByText('Archived conversation')
  expect(screen.queryByText('Live pinned backfill')).toBeNull()
  expect(screen.queryByText('Bot Chat')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Done' }))
  fireEvent.click(screen.getByRole('checkbox', { name: 'Select visible sessions' }))
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
  const confirmation = await screen.findByRole('dialog')
  expect(within(confirmation).getByText('Archived conversation')).toBeTruthy()
  expect(within(confirmation).queryByText('Live pinned backfill')).toBeNull()
  fireEvent.click(within(confirmation).getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
})

it('removes applied filter chips independently without resetting the other filters', async () => {
  renderHistory()
  await screen.findByText('History entry 24')
  fireEvent.click(screen.getByRole('button', { name: 'Filter' }))
  fireEvent.click(screen.getByRole('checkbox', { name: 'Schedule' }))
  fireEvent.click(screen.getByRole('radio', { name: 'Include' }))
  fireEvent.click(screen.getByRole('radio', { name: 'None' }))
  fireEvent.click(screen.getByRole('button', { name: 'Done' }))
  fireEvent.click(screen.getByRole('button', { name: 'Source: Schedule' }))
  expect(screen.queryByRole('button', { name: 'Source: Schedule' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Archived: Include' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Group by: None' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Archived: Include' }))
  expect(screen.queryByRole('button', { name: 'Archived: Include' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Group by: None' }))
  expect(screen.queryByRole('button', { name: 'Group by: None' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Filter' })).toBeTruthy()
})

it('shows partial selection and selects or clears the visible history together', async () => {
  renderHistory()
  await screen.findByText('History entry 24')
  const all = screen.getByRole('checkbox', { name: 'Select visible sessions' })
  const row = screen.getByRole('checkbox', { name: 'Selected: History entry 0' })
  fireEvent.click(row)
  expect(all.getAttribute('aria-checked')).toBe('mixed')
  fireEvent.click(all)
  expect(all.getAttribute('aria-checked')).toBe('true')
  expect(row.getAttribute('aria-checked')).toBe('true')
  fireEvent.click(all)
  expect(all.getAttribute('aria-checked')).toBe('false')
  expect(row.getAttribute('aria-checked')).toBe('false')
})

it('previews clean-up live as the options change and deletes only the preview of the current options', async () => {
  const candidate = makeSessionInfo({ id: 'old', title: 'Old conversation', message_count: 3 })
  const empty = makeSessionInfo({ id: 'empty', title: 'Empty conversation', message_count: 0 })
  const live = makeSessionInfo({ id: 'running', title: 'Running conversation' })

  let release: () => void = () => {}
  vi.mocked(previewHistoryCleanup).mockImplementation(async (_owner, criteria) => {
    if (criteria.messages === 'empty') {
      await new Promise<void>(resolve => (release = resolve))
    }

    return {
      candidates: criteria.messages === 'empty' ? [empty] : [candidate],
      running: [live],
      criteria,
      now: Date.now() / 1000
    }
  })
  vi.mocked(deleteHistorySelection).mockResolvedValue({ deleted: [], running: ['empty'], changed: [], errors: [] })
  const onDeleted = vi.fn()
  const onClose = vi.fn()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  clients.push(client)
  render(
    <QueryClientProvider client={client}>
      <HistoryCleanupDialog
        current={() => true}
        onClose={onClose}
        onDeleted={onDeleted}
        owner={{ connectionId: 'alpha', profile: 'research' }}
      />
    </QueryClientProvider>
  )
  // The dry run starts with the dialog: no separate preview step.
  await screen.findByText('Old conversation')
  expect(screen.getByText('1 item will be deleted')).toBeTruthy()
  expect(screen.getByText('Running conversation')).toBeTruthy()
  expect(screen.getByText('Running · skipped')).toBeTruthy()
  expect(deleteHistorySelection).not.toHaveBeenCalled()

  fireEvent.click(screen.getByRole('button', { name: 'Empty only' }))
  // While the new options are counted the old list stays, but it can no longer be confirmed.
  await waitFor(() => expect(previewHistoryCleanup).toHaveBeenCalledTimes(2))
  expect(screen.getByRole('button', { name: 'Delete 1 item' }).hasAttribute('disabled')).toBe(true)
  release()
  await screen.findByText('Empty conversation')
  expect(screen.queryByText('Old conversation')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Delete 1 item' }))
  await waitFor(() =>
    expect(onDeleted).toHaveBeenCalledWith({ deleted: [], running: ['running', 'empty'], changed: [], errors: [] })
  )
  expect(deleteHistorySelection).toHaveBeenCalledWith(
    { connectionId: 'alpha', profile: 'research' },
    [empty],
    expect.any(Function),
    expect.any(String),
    expect.objectContaining({ criteria: expect.objectContaining({ messages: 'empty' }) })
  )
  expect(onClose).toHaveBeenCalledOnce()
})

it('titles the delete confirmation by what will be deleted and names the running item it skips', async () => {
  vi.mocked(deleteHistorySelection).mockResolvedValue({ deleted: ['history-1'], running: [], changed: [], errors: [] })
  renderHistory()
  await screen.findByText('History entry 24')
  fireEvent.click(screen.getByRole('checkbox', { name: 'Selected: History entry 0' }))
  fireEvent.click(screen.getByRole('checkbox', { name: 'Selected: History entry 1' }))
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
  const confirmation = await screen.findByRole('dialog')
  expect(within(confirmation).getByRole('heading', { name: 'Delete 2 items?' })).toBeTruthy()
  expect(within(confirmation).getAllByText('Chat · 0 messages')).toHaveLength(2)
  expect(within(confirmation).queryByText(/still running/)).toBeNull()
  expect(within(confirmation).queryByRole('button', { name: 'Close' })).toBeNull()
  fireEvent.click(within(confirmation).getByRole('button', { name: 'Delete 2 items' }))
  await waitFor(() => expect(deleteHistorySelection).toHaveBeenCalledOnce())
})

it.each([
  ['Pin', { pinned: true }],
  ['Archive', { archived: true, pinned: false }]
] as const)('applies bulk %s only to the selected owner-qualified sessions', async (action, patch) => {
  vi.mocked(mutateHistorySession).mockResolvedValue({ ok: true })
  renderHistory()
  await screen.findByText('History entry 24')
  fireEvent.click(screen.getByRole('checkbox', { name: 'Selected: History entry 0' }))
  fireEvent.click(screen.getByRole('checkbox', { name: 'Selected: History entry 1' }))
  fireEvent.click(screen.getByRole('button', { name: action }))
  await waitFor(() => expect(mutateHistorySession).toHaveBeenCalledTimes(2))
  expect(
    vi.mocked(mutateHistorySession).mock.calls.map(([owner, id, method, payload]) => ({ owner, id, method, payload }))
  ).toEqual(
    ['history-0', 'history-1'].map(id => ({
      owner: { connectionId: 'alpha', profile: 'research' },
      id,
      method: 'PATCH',
      payload: patch
    }))
  )
})
