/**
 * Where the board switcher mounts (#123597). The full page projects it into
 * the workspace page header; a route tile has no painted page header, so the
 * same switcher sits in the board's own header row. These run the REAL
 * registry, `Contribute`, `Slot` and `RouteTilePane` with only the REST
 * layer mocked.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The placement under test lives in core: the route tile, the page-header host
// and the Slot that reads the area. Plugins can't reach those at runtime.
// eslint-disable-next-line no-restricted-imports
import { RouteTilePane } from '@/app/chat/route-tile'
// eslint-disable-next-line no-restricted-imports
import { WorkspacePageHeaderHostContext } from '@/app/contrib/workspace-page-header'
// eslint-disable-next-line no-restricted-imports
import { WORKSPACE_PAGE_HEADER_AREA } from '@/app/routes'
// eslint-disable-next-line no-restricted-imports
import { Slot } from '@/contrib/react/slot'
// eslint-disable-next-line no-restricted-imports
import { registry } from '@/contrib/registry'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug } from './api'
import { KanbanBoardPage } from './board'
import { KANBAN_LOCALES } from './i18n'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoards: vi.fn(async () => ({
    boards: [
      { name: 'Shipping', project_id: null, slug: 'shipping', total: 3 },
      { name: 'Research', project_id: null, slug: 'research', total: 1 }
    ],
    current: 'shipping'
  })),
  fetchBoard: vi.fn(async () => ({ assignees: [], columns: [], tenants: [] })),
  fetchOrchestration: vi.fn(async () => ({ default_assignee: '' })),
  fetchProfiles: vi.fn(async () => ({ profiles: [] }))
}))

// The trigger's accessible name, built from the loaded en strings
// (`${k.board}: ${label}`). Exact, so a copy change fails loudly.
const SWITCHER = 'Board: Shipping'

let disposeLocales: () => void = () => undefined
let disposePage: () => void = () => undefined
let queryClient = new QueryClient()

beforeEach(() => {
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposePage = registry.register({
    area: 'routes',
    id: 'kanban:page',
    data: { path: '/kanban' },
    render: () => <KanbanBoardPage />
  })
})

afterEach(() => {
  cleanup()
  disposePage()
  disposeLocales()
  $boardSlug.set('')
  vi.restoreAllMocks()
})

const withQuery = (ui: ReactNode) => <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>

// Another page's (or this page's) painted workspace header: the one Slot that
// reads the area, as the workspace zone renders it.
const pageHeader = () => (
  <div data-testid="page-header">
    <Slot area={WORKSPACE_PAGE_HEADER_AREA} />
  </div>
)

// The full page: the workspace pane's routes, inside the host provider.
const workspacePage = () => (
  <div data-testid="workspace">
    <WorkspacePageHeaderHostContext.Provider value={true}>
      <KanbanBoardPage />
    </WorkspacePageHeaderHostContext.Provider>
  </div>
)

const tile = () => (
  <div data-testid="tile">
    <RouteTilePane path="/kanban" />
  </div>
)

const switchers = (scope: HTMLElement) => within(scope).queryAllByRole('button', { name: SWITCHER })
const boardHeader = (scope: HTMLElement) => within(scope).getByRole('banner')
const switcherEntries = () => registry.getArea(WORKSPACE_PAGE_HEADER_AREA).filter(c => c.id === 'kanban:board-switcher')

describe('kanban board switcher placement (#123597)', () => {
  it('a split tile shows the switcher in the board header and contributes nothing to the page header', async () => {
    const register = vi.spyOn(registry, 'register')

    render(withQuery(tile()))

    const inTile = screen.getByTestId('tile')

    await within(boardHeader(inTile)).findByRole('button', { name: SWITCHER })
    expect(switchers(boardHeader(inTile))).toHaveLength(1)
    expect(registry.getArea(WORKSPACE_PAGE_HEADER_AREA)).toHaveLength(0)
    expect(register.mock.calls.some(([c]) => c.area === WORKSPACE_PAGE_HEADER_AREA)).toBe(false)
  })

  it('the full page keeps the switcher in the page header, not the board header', async () => {
    render(
      withQuery(
        <>
          {pageHeader()}
          {workspacePage()}
        </>
      )
    )

    const header = screen.getByTestId('page-header')

    await within(header).findByRole('button', { name: SWITCHER })
    expect(switchers(header)).toHaveLength(1)
    expect(switchers(boardHeader(screen.getByTestId('workspace')))).toHaveLength(0)
    expect(screen.getAllByRole('button', { name: SWITCHER })).toHaveLength(1)
  })

  it("a tile's switcher does not leak into an available page-header slot", async () => {
    render(
      withQuery(
        <>
          {pageHeader()}
          {tile()}
        </>
      )
    )

    const inTile = screen.getByTestId('tile')

    await within(boardHeader(inTile)).findByRole('button', { name: SWITCHER })
    expect(switchers(boardHeader(inTile))).toHaveLength(1)
    expect(switchers(screen.getByTestId('page-header'))).toHaveLength(0)
  })

  it('the full page and a tile each keep one switcher, and closing the tile leaves the page its own', async () => {
    const view = render(
      withQuery(
        <>
          {pageHeader()}
          {workspacePage()}
          {tile()}
        </>
      )
    )

    const header = screen.getByTestId('page-header')

    await within(header).findByRole('button', { name: SWITCHER })
    await within(boardHeader(screen.getByTestId('tile'))).findByRole('button', { name: SWITCHER })
    expect(switchers(header)).toHaveLength(1)
    expect(switchers(screen.getByTestId('tile'))).toHaveLength(1)
    expect(switcherEntries()).toHaveLength(1)

    view.rerender(
      withQuery(
        <>
          {pageHeader()}
          {workspacePage()}
        </>
      )
    )

    await waitFor(() => expect(screen.queryByTestId('tile')).toBeNull())
    expect(switchers(screen.getByTestId('page-header'))).toHaveLength(1)
    expect(switcherEntries()).toHaveLength(1)
  })
})
