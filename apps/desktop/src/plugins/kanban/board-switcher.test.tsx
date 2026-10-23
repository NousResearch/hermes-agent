import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug } from './api'
import { BoardSwitcher } from './board-switcher'
import { $unseenByBoard, cursorKey } from './completion-notify'
import { KANBAN_LOCALES } from './i18n'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoards: vi.fn(async () => ({
    boards: [
      { name: 'Shipping', project_id: null, slug: 'shipping', total: 3 },
      { name: 'Research', project_id: null, slug: 'research', total: 9 }
    ],
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
  $unseenByBoard.set({})
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

  // #123596: per-board unseen terminal events, in the board list.
  it('shows an unseen count for a board with unseen events and none at zero', async () => {
    $unseenByBoard.set({ [cursorKey('local', 'research')]: 4, [cursorKey('local', 'shipping')]: 0 })
    mount()

    const trigger = await screen.findByRole('button', { name: /Shipping/ })
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })

    const research = await screen.findByRole('menuitem', { name: /Research/ })
    const shipping = screen.getByRole('menuitem', { name: /Shipping/ })

    expect(within(research).getByText('4')).toBeTruthy()
    // Only the card total, no unseen badge.
    expect(within(shipping).queryByText('0')).toBeNull()
    expect(shipping.querySelector('[data-slot="badge"]')).toBeNull()
  })
})
