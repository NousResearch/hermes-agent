import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug } from './api'
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

  // The trigger is projected into the Kanban page header as the board's own name, so it
  // must announce itself as a control: a visible "Board" label, a "Board: …"
  // accessible name, and a "Switch board" tooltip on hover.
  it('identifies the current board switcher as a control', async () => {
    mount()

    const trigger = await screen.findByRole('button', { name: 'Board: Shipping' })

    expect(trigger.textContent).toContain('Board')

    fireEvent.pointerMove(trigger, { pointerType: 'mouse' })

    expect((await screen.findByRole('tooltip')).textContent).toContain('Switch board')
  })
})
