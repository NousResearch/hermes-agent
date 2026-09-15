import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { $boardSlug } from './api'
import { KanbanBoardPage } from './board'
import { BoardSwitcher } from './board-switcher'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoard: vi.fn(async () => ({ columns: [], assignees: [], tenants: [], latest_event_id: 0, now: 0 })),
  fetchProfiles: vi.fn(async () => ({ profiles: [] })),
  fetchBoards: vi.fn(async () => ({
    boards: [{ name: 'Shipping', project_id: null, slug: 'shipping', total: 3 }],
    current: 'shipping'
  }))
}))

afterEach(() => {
  cleanup()
  $boardSlug.set('')
})

const mount = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <BoardSwitcher />
    </QueryClientProvider>
  )

describe('board switcher', () => {
  it('keeps the board picker inside the page header, away from sidebar tabs', async () => {
    const { container } = render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <KanbanBoardPage />
      </QueryClientProvider>
    )

    const picker = await screen.findByText('Shipping')

    expect(picker.closest('header')).toBe(container.querySelector('header'))
    expect(container.querySelector('header')).not.toBeNull()
  })

  // The rename and settings dialogs stay mounted while closed, so they render
  // with a null board on every pass. Reading the slug inside their mutation
  // callback used to crash the whole contribution, because the React Compiler
  // lifts a callback's property reads into its render-time dependency check.
  it('renders while its dialogs are closed', async () => {
    mount()

    expect(await screen.findByText('Shipping')).toBeTruthy()
  })
})
