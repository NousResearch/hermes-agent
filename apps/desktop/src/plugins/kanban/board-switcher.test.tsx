import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug, fetchBoards } from './api'
import { BoardSwitcher } from './board-switcher'
import { KANBAN_LOCALES } from './i18n'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoards: vi.fn(async () => ({
    boards: [{ name: 'Shipping', project_id: null, slug: 'shipping', total: 3 }],
    current: 'shipping'
  }))
}))

let disposeLocales: () => void = () => undefined

beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  $boardSlug.set('')
})

const mount = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <BoardSwitcher />
    </QueryClientProvider>
  )

describe('board switcher', () => {
  // The rename and settings dialogs stay mounted while closed, so they render
  // with a null board on every pass. Reading the slug inside their mutation
  // callback used to crash the whole contribution, because the React Compiler
  // lifts a callback's property reads into its render-time dependency check.
  it('renders while its dialogs are closed', async () => {
    mount()

    expect(await screen.findByText('Shipping')).toBeTruthy()
  })

  // Pause is per board: the trigger flags the CURRENT board, and the menu
  // flags each held one — a single unqualified "paused" would be a lie as soon
  // as two boards exist (#141451).
  it('flags the current board in the trigger and each held board in the menu', async () => {
    vi.mocked(fetchBoards).mockResolvedValue({
      boards: [
        { name: 'Shipping', pause: { board: 'shipping', paused_at: 'x', reason: null }, slug: 'shipping', total: 3 },
        { name: 'Billing', pause: null, slug: 'billing', total: 1 },
        { name: 'Hold', pause: { board: 'hold', paused_at: 'x', reason: 'provider swap' }, slug: 'hold', total: 0 }
      ],
      current: 'shipping'
    })
    mount()

    expect(await screen.findByTestId('kanban-switcher-paused')).toBeTruthy()

    const trigger = screen.getByRole('button', { name: /Board: Shipping/ })
    fireEvent.pointerDown(trigger)
    fireEvent.click(trigger)

    const held = await screen.findByTestId('kanban-board-paused-shipping')
    expect(held.textContent).toContain('PAUSED')
    expect(await screen.findByTestId('kanban-board-paused-hold')).toBeTruthy()
    expect(screen.queryByTestId('kanban-board-paused-billing')).toBeNull()
  })
})
