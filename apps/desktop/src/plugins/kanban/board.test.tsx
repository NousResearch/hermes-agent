import { host } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug, boardsKey, fetchBoards } from './api'
import { KanbanBoardPage } from './board'
import { en, KANBAN_LOCALES } from './i18n'
import plugin from './plugin'
import type { BoardsResponse, KanbanBoard, KanbanTask } from './types'
import { $boardRequest } from './ui'

vi.mock('@/hermes', () => ({ getGlobalModelOptions: vi.fn(), setApiRequestProfile: vi.fn() }))

const META_LINE =
  '> Fleet: revision 3 | point Conductor: tb-cndr | campaign: none | repository: none | canonical status: todo'

// A card exactly as the fleet sync adapter presents it locally: a `[Sync …]`
// title prefix plus the `fleet-kanban:meta` block on top of the real body. The
// node rides in `tenant` (the adapter's `current_node`).
const decorated: KanbanTask = {
  id: 't_fk_43264da9',
  title: '[Sync pending] Rotate the turnerbook canary',
  body: `<!-- fleet-kanban:meta -->\n${META_LINE}\n<!-- /fleet-kanban:meta -->\n\nRun the rotation from the node itself.`,
  status: 'todo',
  assignee: 'tb-cndr',
  tenant: 'turnerbook',
  created_at: 1_789_799_466
}

// A manual status change leaves the backend's administrative note in
// `latest_summary` — it says nothing about the work.
const adminNoted: KanbanTask = {
  id: 't_fk_80209e98',
  title: 'Drain the sheldon queue',
  body: 'Readable drain notes.',
  latest_summary: 'status changed to todo (dashboard/direct)',
  status: 'todo',
  assignee: 'shld-cndr',
  tenant: 'sheldon'
}

const summarised: KanbanTask = {
  id: 't_fk_22aaf3f1',
  title: 'Rotate the snowdrop canary',
  body: 'Readable snowdrop notes.',
  latest_summary: 'Rotated the canary; PR #12 opened.',
  status: 'done',
  assignee: 'snow-cndr',
  tenant: 'snowdrop'
}

const STATUSES = ['triage', 'todo', 'scheduled', 'ready', 'running', 'blocked', 'review', 'done']

const boardWith = (tasks: KanbanTask[]): KanbanBoard => ({
  columns: STATUSES.map(name => ({ name, tasks: tasks.filter(task => task.status === name) })),
  tenants: [...new Set(tasks.flatMap(task => (task.tenant ? [task.tenant] : [])))],
  assignees: [...new Set(tasks.flatMap(task => (task.assignee ? [task.assignee] : [])))],
  latest_event_id: 1,
  now: 1_789_800_000
})

const fleetBoard = boardWith([decorated, adminNoted, summarised])
// The same decorated row on an ORDINARY board, where it is just a title, a body and a tenant.
const ordinaryBoard = boardWith([decorated])

const boards: BoardsResponse = {
  boards: [
    { is_current: true, name: 'Default', slug: 'default', total: 0 },
    { is_current: false, name: 'Fleet', slug: 'fleet', total: 3 },
    { is_current: false, name: 'Shipping', slug: 'shipping', total: 1 }
  ],
  current: 'default'
}

const BOARDS_KEY = boardsKey('local')

// Per-test overrides (reset after each test).
let boardsResponse = boards
/** Which board each board fetch was bound to, in order. */
const fetchedBoards: string[] = []

vi.mock('./api', async importOriginal => {
  const api = await importOriginal<typeof KanbanApi>()

  return {
    ...api,
    fetchBoard: vi.fn(async (_archived: boolean, slug?: string) => {
      const bound = slug ?? api.$boardSlug.get()

      fetchedBoards.push(bound)

      return bound === 'fleet' || (bound === '' && boardsResponse.current === 'fleet') ? fleetBoard : ordinaryBoard
    }),
    fetchBoards: vi.fn(async () => boardsResponse),
    fetchOrchestration: vi.fn(async () => ({
      auto_decompose: false,
      default_assignee: '',
      orchestrator_profile: '',
      resolved_default_assignee: '',
      resolved_orchestrator_profile: ''
    })),
    fetchProfiles: vi.fn(async () => ({ profiles: [] }))
  }
})

type Ctx = Parameters<typeof plugin.register>[0]
type Registered = { area: string; data?: unknown; id: string }

/** Register the plugin against a minimal host and hand back its "Open Fleet
 *  board" command — the entry an operator actually uses. */
function registerPlugin() {
  const contributions: Registered[] = []
  const disposers: Array<() => void> = []

  plugin.register({
    i18n: { onLocaleChange: vi.fn(() => vi.fn()), register: vi.fn(), t: (key: string) => key },
    onDispose: (dispose: () => void) => disposers.push(dispose),
    os: undefined,
    registerMany: (items: Registered[]) => contributions.push(...items),
    rest: vi.fn(),
    socket: () => vi.fn(),
    storage: { get: (_key: string, fallback: unknown) => fallback, remove: vi.fn(), set: vi.fn() }
  } as unknown as Ctx)

  const row = contributions.find(item => item.id === 'open-fleet')?.data as { run: () => void }

  return { dispose: () => disposers.forEach(fn => fn()), openFleet: () => act(() => row.run()) }
}

let client: QueryClient
let disposeLocales: () => void = () => undefined
let disposePlugin: () => void = () => undefined
let openFleet: () => void = () => undefined
let navigate: ReturnType<typeof vi.spyOn>

/** Hold the next board-list response until the test releases it. */
function deferBoards() {
  let release: () => void = () => undefined

  vi.mocked(fetchBoards).mockImplementationOnce(
    () =>
      new Promise<BoardsResponse>(resolve => {
        release = () => resolve(boardsResponse)
      })
  )

  return () => act(async () => release())
}

const settle = () => act(() => new Promise(resolve => setTimeout(resolve, 50)))

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  ;({ dispose: disposePlugin, openFleet } = registerPlugin())
  // The router's side of host.navigate, without the app shell: the hash moves.
  navigate = vi.spyOn(host, 'navigate').mockImplementation(to => {
    window.location.hash = `#${to}`
  })
  fetchedBoards.length = 0
})

afterEach(() => {
  cleanup()
  client.clear()
  disposePlugin()
  disposeLocales()
  navigate.mockRestore()
  $boardSlug.set('')
  $boardRequest.set(null)
  boardsResponse = boards
  window.location.hash = ''
  vi.resetAllMocks()
})

const mount = () =>
  render(
    <QueryClientProvider client={client}>
      <KanbanBoardPage />
    </QueryClientProvider>
  )

const cardOf = (element: HTMLElement) => within(element.closest<HTMLElement>('[draggable="true"]')!)

describe('fleet card face', () => {
  it('shows a readable title, body, status, owner and node — never the sync metadata', async () => {
    $boardSlug.set('fleet')
    mount()

    const face = cardOf(await screen.findByText('Rotate the turnerbook canary'))

    expect(face.getByText('Run the rotation from the node itself.')).toBeTruthy()
    expect(face.getByText('Todo')).toBeTruthy()
    expect(face.getByText('tb-cndr')).toBeTruthy()
    expect(face.getByText('turnerbook')).toBeTruthy()

    // Internal synchronization bookkeeping stays off the face.
    expect(screen.queryByText(/fleet-kanban:meta/)).toBeNull()
    expect(screen.queryByText(/revision 3/)).toBeNull()
    expect(screen.queryByText(/Sync pending/)).toBeNull()
  })

  it('hides the administrative note a manual status change leaves behind, keeps real summaries', async () => {
    $boardSlug.set('fleet')
    mount()

    const drain = cardOf(await screen.findByText('Drain the sheldon queue'))

    expect(drain.getByText('Readable drain notes.')).toBeTruthy()
    expect(screen.queryByText('status changed to todo (dashboard/direct)')).toBeNull()

    const rotate = cardOf(screen.getByText('Rotate the snowdrop canary'))

    expect(rotate.getByText('Rotated the canary; PR #12 opened.')).toBeTruthy()
  })
})

describe('ordinary boards stay literal', () => {
  it('shows an ordinary board’s title, body and tenant exactly as stored', async () => {
    $boardSlug.set('shipping')
    mount()

    const face = cardOf(await screen.findByText('[Sync pending] Rotate the turnerbook canary'))

    // The body is the body, the tenant is a tenant — not a fleet node.
    expect(face.getByText(/fleet-kanban:meta/)).toBeTruthy()
    expect(face.queryByText('turnerbook')).toBeNull()
    expect(face.getByText('Todo')).toBeTruthy()
    expect(face.getByText('tb-cndr')).toBeTruthy()
  })

  it('verifies which board the server’s current one is before painting it', async () => {
    // Nothing selected: the server's current board decides, and that is only
    // known once the board list answers.
    boardsResponse = {
      boards: [
        { is_current: true, name: 'Fleet', slug: 'fleet', total: 3 },
        { is_current: false, name: 'Shipping', slug: 'shipping', total: 1 }
      ],
      current: 'fleet'
    }
    const release = deferBoards()

    mount()

    await waitFor(() => expect(fetchedBoards).toEqual(['']))
    await settle()
    expect(screen.queryByText('Rotate the turnerbook canary')).toBeNull()
    expect(screen.queryByText('[Sync pending] Rotate the turnerbook canary')).toBeNull()

    await release()

    expect(await screen.findByText('Rotate the turnerbook canary')).toBeTruthy()
  })
})

describe('fleet-scoped entry', () => {
  it('shows nothing actionable and binds no board request until the requested board resolves', async () => {
    $boardSlug.set('shipping')
    const release = deferBoards()

    openFleet()
    mount()

    const newTask = await screen.findByRole('button', { name: 'New task' })

    expect(newTask).toHaveProperty('disabled', true)
    await settle()
    expect(screen.queryByText('[Sync pending] Rotate the turnerbook canary')).toBeNull()
    expect(fetchedBoards).toEqual([])

    await release()

    await waitFor(() => expect($boardSlug.get()).toBe('fleet'))
    expect(await screen.findByText('Rotate the turnerbook canary')).toBeTruthy()
    expect(fetchedBoards).toEqual(['fleet'])
    expect(screen.getByRole('button', { name: 'New task' })).toHaveProperty('disabled', false)
  })

  it('enters again when the command runs after a manual switch, and a later manual switch stands', async () => {
    openFleet()
    mount()

    await waitFor(() => expect($boardSlug.get()).toBe('fleet'))

    // The operator switches boards by hand…
    act(() => $boardSlug.set('shipping'))
    await screen.findByText('[Sync pending] Rotate the turnerbook canary')

    // …and runs the command again.
    openFleet()
    await waitFor(() => expect($boardSlug.get()).toBe('fleet'))

    // A manual choice AFTER the entry survives a board-list refresh.
    act(() => $boardSlug.set('shipping'))
    await act(() => client.invalidateQueries({ queryKey: BOARDS_KEY }))
    await settle()
    expect($boardSlug.get()).toBe('shipping')
  })

  it('keeps the operator’s own selection on a plain entry', async () => {
    $boardSlug.set('shipping')
    mount()

    await screen.findByText('[Sync pending] Rotate the turnerbook canary')

    expect($boardSlug.get()).toBe('shipping')
  })

  it('leaves the selection alone, and says so, when the requested board is not on this backend', async () => {
    const notify = vi.spyOn(host, 'notify').mockImplementation(() => '')

    $boardSlug.set('shipping')
    boardsResponse = { boards: boards.boards.filter(meta => meta.slug !== 'fleet'), current: 'default' }

    openFleet()
    mount()

    expect(await screen.findByText('[Sync pending] Rotate the turnerbook canary')).toBeTruthy()
    expect($boardSlug.get()).toBe('shipping')
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'warning' }))

    notify.mockRestore()
  })
})

describe('fleet-scoped entry — rejected validation', () => {
  it('shows the failure and keeps everything gated until a retry succeeds', async () => {
    $boardSlug.set('shipping')
    vi.mocked(fetchBoards).mockRejectedValueOnce(new Error('503: {"detail":"boards unavailable"}'))

    openFleet()
    mount()
    await settle()

    // Shipping never becomes actionable: no New task, no cards, no board fetch.
    expect(screen.getByRole('button', { name: 'New task' })).toHaveProperty('disabled', true)
    expect(screen.queryByText('[Sync pending] Rotate the turnerbook canary')).toBeNull()
    expect(fetchedBoards).toEqual([])
    expect($boardSlug.get()).toBe('shipping')

    // The validation failure is what the operator sees.
    expect(await screen.findByText(en.boardsCheckFailed('fleet'))).toBeTruthy()
    expect(screen.getByText('boards unavailable')).toBeTruthy()

    // Retry: the list answers, the entry resolves, the fleet board is bound.
    fireEvent.click(screen.getByRole('button', { name: en.retry }))

    await waitFor(() => expect($boardSlug.get()).toBe('fleet'))
    expect(await screen.findByText('Rotate the turnerbook canary')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'New task' })).toHaveProperty('disabled', false)
    expect(fetchedBoards).toEqual(['fleet'])
  })

  it('returns to the operator’s own board only when they cancel the entry', async () => {
    $boardSlug.set('shipping')
    vi.mocked(fetchBoards).mockRejectedValueOnce(new Error('boards unavailable'))

    openFleet()
    mount()
    await settle()

    expect(screen.getByRole('button', { name: 'New task' })).toHaveProperty('disabled', true)
    expect(screen.queryByText('[Sync pending] Rotate the turnerbook canary')).toBeNull()
    expect(await screen.findByText(en.boardsCheckFailed('fleet'))).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: en.cancel }))

    expect(await screen.findByText('[Sync pending] Rotate the turnerbook canary')).toBeTruthy()
    expect($boardSlug.get()).toBe('shipping')
    expect(screen.getByRole('button', { name: 'New task' })).toHaveProperty('disabled', false)
  })
})

describe('fleet-scoped entry — stale cache, rejected fresh validation', () => {
  it('never consumes the request on cached data: the rejected refetch gates the entry', async () => {
    // Prime the board list, then let it go stale.
    $boardSlug.set('shipping')
    mount()
    await screen.findByText('[Sync pending] Rotate the turnerbook canary')
    await act(() => client.invalidateQueries({ queryKey: BOARDS_KEY, refetchType: 'none' }))
    vi.mocked(fetchBoards).mockRejectedValueOnce(new Error('503: {"detail":"boards unavailable"}'))

    openFleet()
    await settle()

    // Cached data is not a validation: Shipping stays selected but gated, and no board is bound.
    expect($boardSlug.get()).toBe('shipping')
    expect(screen.getByRole('button', { name: 'New task' })).toHaveProperty('disabled', true)
    expect(screen.queryByText('[Sync pending] Rotate the turnerbook canary')).toBeNull()
    expect(fetchedBoards).toEqual(['shipping'])
    expect(await screen.findByText(en.boardsCheckFailed('fleet'))).toBeTruthy()

    // Only a fresh, successful validation resolves it.
    fireEvent.click(screen.getByRole('button', { name: en.retry }))

    await waitFor(() => expect($boardSlug.get()).toBe('fleet'))
    expect(await screen.findByText('Rotate the turnerbook canary')).toBeTruthy()
    expect(fetchedBoards).toEqual(['shipping', 'fleet'])
  })

  it('gates a re-entered page the same way while its background refetch rejects', async () => {
    $boardSlug.set('shipping')
    const view = mount()

    await screen.findByText('[Sync pending] Rotate the turnerbook canary')
    view.unmount()
    await act(() => client.invalidateQueries({ queryKey: BOARDS_KEY, refetchType: 'none' }))
    vi.mocked(fetchBoards).mockRejectedValueOnce(new Error('boards unavailable'))

    openFleet()
    mount()
    await settle()

    expect($boardSlug.get()).toBe('shipping')
    expect(screen.getByRole('button', { name: 'New task' })).toHaveProperty('disabled', true)
    expect(screen.queryByText('[Sync pending] Rotate the turnerbook canary')).toBeNull()
    expect(await screen.findByText(en.boardsCheckFailed('fleet'))).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: en.cancel }))

    expect(await screen.findByText('[Sync pending] Rotate the turnerbook canary')).toBeTruthy()
    expect($boardSlug.get()).toBe('shipping')
  })
})
