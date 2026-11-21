/**
 * `/kanban?board=<slug>&task=<id>` — the deep link other plugins (and the
 * completion toast) use to open a board with one card's drawer showing. The
 * REAL page, router, plugin storage and connection store run; only the REST
 * layer and the drawer body are stubbed.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The real namespaced plugin storage, so persistence runs its production path.
// eslint-disable-next-line no-restricted-imports
import { createPluginContext } from '@/contrib/plugin'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'
// Test harness reads the host's toast stack.
// eslint-disable-next-line no-restricted-imports
import { $notifications, clearNotifications } from '@/store/notifications'
// eslint-disable-next-line no-restricted-imports
import { setConnection } from '@/store/session'

import type * as KanbanApi from './api'
import { $boardSlug, bindApi } from './api'
import { KanbanBoardPage } from './board'
import { KANBAN_LOCALES } from './i18n'
import { kanbanHref } from './route'

const { boards } = vi.hoisted(() => {
  const card = (id: string, status = 'ready') => ({ id, status, title: `Card ${id}` })

  return {
    boards: {
      research: { assignees: [], columns: [{ name: 'ready', tasks: [card('t_r1')] }], tenants: [] },
      shipping: { assignees: [], columns: [{ name: 'ready', tasks: [card('t_s1')] }], tenants: [] }
    } as Record<string, unknown>
  }
})

vi.mock('./api', async importOriginal => {
  const actual = await importOriginal<typeof KanbanApi>()

  return {
    ...actual,
    fetchBoards: vi.fn(async () => ({
      boards: [
        { name: 'Shipping', project_id: null, slug: 'shipping', total: 1 },
        { name: 'Research', project_id: null, slug: 'research', total: 1 }
      ],
      current: 'shipping'
    })),
    // '' is the server's current board, as the real `?board=`-less request.
    fetchBoard: vi.fn(async () => boards[actual.$boardSlug.get() || 'shipping']),
    fetchOrchestration: vi.fn(async () => ({ default_assignee: '' })),
    fetchProfiles: vi.fn(async () => ({ profiles: [] }))
  }
})

// The drawer's own data and layout have their own suite; here only WHICH card
// it was asked to show matters.
vi.mock('./drawer', () => ({
  TaskDrawer: ({ id }: { id: null | string }) => (id ? <div aria-label={`Drawer ${id}`} role="dialog" /> : null)
}))

let disposeLocales: () => void = () => undefined
let disposeApi: () => void = () => undefined

beforeEach(() => {
  localStorage.clear()
  setConnection(null)
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeApi()
  disposeApi = () => undefined
  disposeLocales()
  clearNotifications()
  setConnection(null)
  $boardSlug.set('')
})

function bindPersistence() {
  const ctx = createPluginContext('kanban')

  disposeApi = bindApi(
    async () => ({ current: 'shipping', latest_event_id: 0 }) as never,
    ctx.storage,
    () => () => {}
  )

  return ctx.storage
}

function Location() {
  const location = useLocation()

  return <output aria-label="location">{`${location.pathname}${location.search}`}</output>
}

function mount(to: string, page: ReactNode = <KanbanBoardPage />) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[to]}>
        {page}
        <Location />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

const location = () => screen.getByRole('status', { name: 'location' }).textContent
const drawers = () => screen.queryAllByRole('dialog')
const messages = () => $notifications.get().map(item => item.message)

describe('kanban deep link', () => {
  it('builds the link a caller navigates to', () => {
    expect(kanbanHref({ board: 'research', task: 't_r1' })).toBe('/kanban?board=research&task=t_r1')
    expect(kanbanHref({ task: 't_s1' })).toBe('/kanban?task=t_s1')
    expect(kanbanHref()).toBe('/kanban')
  })

  it('opens the named board with the card drawer showing, then drops the query', async () => {
    bindPersistence()
    mount(kanbanHref({ board: 'research', task: 't_r1' }))

    expect(await screen.findByRole('dialog', { name: 'Drawer t_r1' })).toBeTruthy()
    expect($boardSlug.get()).toBe('research')
    expect(location()).toBe('/kanban')
    expect(messages()).toEqual([])
  })

  it('a task alone opens on the selected board, and the server-current board selects as ""', async () => {
    bindPersistence()
    $boardSlug.set('research')
    mount(kanbanHref({ board: 'shipping', task: 't_s1' }))

    expect(await screen.findByRole('dialog', { name: 'Drawer t_s1' })).toBeTruthy()
    expect($boardSlug.get()).toBe('')

    cleanup()
    $boardSlug.set('research')
    mount(kanbanHref({ task: 't_r1' }))

    expect(await screen.findByRole('dialog', { name: 'Drawer t_r1' })).toBeTruthy()
    expect($boardSlug.get()).toBe('research')
  })

  it('an unknown board keeps the current board, says so, and opens nothing', async () => {
    const storage = bindPersistence()
    $boardSlug.set('research')
    mount(kanbanHref({ board: 'nope', task: 't_r1' }))

    await waitFor(() => expect(messages()).toEqual(['No board “nope” on this connection — showing the current board']))
    expect($boardSlug.get()).toBe('research')
    expect(storage.get('boardSlug', '')).toBe('research')
    expect(drawers()).toHaveLength(0)
    expect(location()).toBe('/kanban')
  })

  it('a task missing from the board shows the board and says so', async () => {
    bindPersistence()
    mount(kanbanHref({ board: 'research', task: 't_gone' }))

    await waitFor(() => expect(messages()).toEqual(['Task t_gone isn’t on this board']))
    expect($boardSlug.get()).toBe('research')
    expect(drawers()).toHaveLength(0)
  })

  it('without a query the page keeps its persisted board and opens nothing', async () => {
    const storage = bindPersistence()
    $boardSlug.set('research')
    mount('/kanban')

    expect(await screen.findByText('Card t_r1')).toBeTruthy()
    expect($boardSlug.get()).toBe('research')
    expect(storage.get('boardSlug', '')).toBe('research')
    expect(drawers()).toHaveLength(0)
    expect(messages()).toEqual([])
  })

  it('persists the linked board under the active connection only', async () => {
    const storage = bindPersistence()
    act(() => setConnection({ connectionId: 'remote-test', mode: 'remote' } as never))
    mount(kanbanHref({ board: 'research', task: 't_r1' }))

    expect(await screen.findByRole('dialog', { name: 'Drawer t_r1' })).toBeTruthy()
    expect(storage.get('boardSlug.remote-test', '')).toBe('research')
    expect(storage.get('boardSlug', 'untouched')).toBe('untouched')
  })

  it('two mounted copies of the page act on the link once', async () => {
    bindPersistence()
    mount(
      kanbanHref({ board: 'research', task: 't_r1' }),
      <>
        <KanbanBoardPage />
        <KanbanBoardPage />
      </>
    )

    expect(await screen.findByRole('dialog', { name: 'Drawer t_r1' })).toBeTruthy()
    expect(drawers()).toHaveLength(1)
  })
})
