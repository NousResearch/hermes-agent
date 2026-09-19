import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug } from './api'
import { KanbanBoardPage } from './board'
import { KANBAN_LOCALES } from './i18n'
import type { BoardsResponse, KanbanBoard, KanbanTask } from './types'

vi.mock('@/hermes', () => ({ getGlobalModelOptions: vi.fn(), setApiRequestProfile: vi.fn() }))

// A Fleet card exactly as the sync adapter presents it locally: a `[Sync …]`
// title prefix plus the `fleet-kanban:meta` block on top of the real body. The
// node rides in `tenant` (the adapter's `current_node`).
const META_LINE =
  '> Fleet: revision 3 | point Conductor: tb-cndr | campaign: none | repository: none | canonical status: todo'

const fleetTask: KanbanTask = {
  id: 't_fk_43264da9',
  title: '[Sync pending] Rotate the turnerbook canary',
  body: `<!-- fleet-kanban:meta -->\n${META_LINE}\n<!-- /fleet-kanban:meta -->\n\nRun the rotation from the node itself.`,
  status: 'todo',
  assignee: 'tb-cndr',
  tenant: 'turnerbook',
  created_at: 1_789_799_466
}

const STATUSES = ['triage', 'todo', 'scheduled', 'ready', 'running', 'blocked', 'review', 'done']

const board: KanbanBoard = {
  columns: STATUSES.map(name => ({ name, tasks: name === 'todo' ? [fleetTask] : [] })),
  tenants: ['turnerbook'],
  assignees: ['tb-cndr'],
  latest_event_id: 1,
  now: 1_789_800_000
}

const boards: BoardsResponse = {
  boards: [
    { is_current: true, name: 'Default', slug: 'default', total: 0 },
    { is_current: false, name: 'Fleet', slug: 'fleet', total: 1 },
    { is_current: false, name: 'Shipping', slug: 'shipping', total: 3 }
  ],
  current: 'default'
}

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoard: vi.fn(async () => board),
  fetchBoards: vi.fn(async () => boards),
  fetchOrchestration: vi.fn(async () => ({
    auto_decompose: false,
    default_assignee: '',
    orchestrator_profile: '',
    resolved_default_assignee: '',
    resolved_orchestrator_profile: ''
  })),
  fetchProfiles: vi.fn(async () => ({ profiles: [] }))
}))

let disposeLocales: () => void = () => undefined

beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  $boardSlug.set('')
  window.location.hash = ''
  vi.clearAllMocks()
})

const mount = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <KanbanBoardPage />
    </QueryClientProvider>
  )

describe('fleet card face', () => {
  it('shows a readable title, body, status, owner and node — never the sync metadata', async () => {
    mount()

    const title = await screen.findByText('Rotate the turnerbook canary')
    const card = title.closest<HTMLElement>('[draggable="true"]')

    expect(card).toBeTruthy()

    const face = within(card!)

    expect(face.getByText('Run the rotation from the node itself.')).toBeTruthy()
    expect(face.getByText('Todo')).toBeTruthy()
    expect(face.getByText('tb-cndr')).toBeTruthy()
    expect(face.getByText('turnerbook')).toBeTruthy()

    // Internal synchronization bookkeeping stays off the face.
    expect(screen.queryByText(/fleet-kanban:meta/)).toBeNull()
    expect(screen.queryByText(/revision 3/)).toBeNull()
    expect(screen.queryByText(/Sync pending/)).toBeNull()
  })
})

describe('fleet-scoped entry', () => {
  it('selects the Fleet board when the surface is entered with ?board=fleet', async () => {
    window.location.hash = '#/kanban?board=fleet'
    mount()

    await waitFor(() => expect($boardSlug.get()).toBe('fleet'))
  })

  it('keeps the operator’s own selection on a plain entry', async () => {
    $boardSlug.set('shipping')
    window.location.hash = '#/kanban'
    mount()

    await screen.findByText('Rotate the turnerbook canary')

    expect($boardSlug.get()).toBe('shipping')
  })

  it('leaves the selection alone when the scoped board is not on this backend', async () => {
    $boardSlug.set('shipping')
    window.location.hash = '#/kanban?board=nowhere'
    mount()

    await screen.findByText('Rotate the turnerbook canary')

    expect($boardSlug.get()).toBe('shipping')
  })
})
