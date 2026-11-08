import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The real namespaced plugin storage and connection store, so persistence and
// per-connection scoping run their production path.
// eslint-disable-next-line no-restricted-imports
import { createPluginContext } from '@/contrib/plugin'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'
// eslint-disable-next-line no-restricted-imports
import { setConnection } from '@/store/session'

import type * as KanbanApi from './api'
import { $boardSlug, bindApi, fetchBoards } from './api'
import { BoardSwitcher } from './board-switcher'
import { KANBAN_LOCALES } from './i18n'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoards: vi.fn(async () => ({
    boards: [
      { name: 'Shipping', project_id: null, slug: 'shipping', total: 3 },
      { name: 'Research', project_id: null, slug: 'research', total: 1 }
    ],
    current: 'shipping'
  }))
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
  setConnection(null)
  $boardSlug.set('')
})

const mount = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <BoardSwitcher />
    </QueryClientProvider>
  )

function bindPersistence() {
  const ctx = createPluginContext('kanban')

  disposeApi = bindApi(
    async () => ({ current: 'shipping', latest_event_id: 0 }) as never,
    ctx.storage,
    () => () => {}
  )

  return ctx.storage
}

async function chooseFromMenu(name: RegExp | string) {
  fireEvent.pointerDown(await screen.findByRole('button', { name: /^Board:/ }), { button: 0, ctrlKey: false })
  fireEvent.click(await screen.findByRole('menuitem', { name }))
}

const tabNames = () =>
  within(screen.getByRole('group', { name: 'Pinned boards' }))
    .getAllByRole('button')
    .map(button => button.textContent)

const tab = (name: string) => within(screen.getByRole('group', { name: 'Pinned boards' })).getByRole('button', { name })

describe('board switcher', () => {
  // The rename and settings dialogs stay mounted while closed, so they render
  // with a null board on every pass. Reading the slug inside their mutation
  // callback used to crash the whole contribution, because the React Compiler
  // lifts a callback's property reads into its render-time dependency check.
  it('renders while its dialogs are closed', async () => {
    mount()

    expect(await screen.findByText('Shipping')).toBeTruthy()
  })

  it('pins boards as tabs that switch, reorder without switching, and survive a reload', async () => {
    const storage = bindPersistence()
    const view = mount()

    await chooseFromMenu('Pin board')
    await chooseFromMenu(/^Research/)
    await chooseFromMenu('Pin board')
    expect(tabNames()).toEqual(['Shipping', 'Research'])
    expect($boardSlug.get()).toBe('research')
    expect(tab('Research').getAttribute('aria-pressed')).toBe('true')

    fireEvent.keyDown(tab('Research'), { altKey: true, key: 'ArrowLeft' })
    expect(tabNames()).toEqual(['Research', 'Shipping'])

    const dataTransfer = { dropEffect: '', effectAllowed: '' }
    fireEvent.dragStart(tab('Research'), { dataTransfer })
    fireEvent.dragOver(tab('Shipping'), { dataTransfer })
    fireEvent.drop(tab('Shipping'), { dataTransfer })
    expect(tabNames()).toEqual(['Shipping', 'Research'])
    expect($boardSlug.get()).toBe('research')
    expect(storage.get('pinnedBoards', [])).toEqual(['shipping', 'research'])

    // The server's current board is selected as '' — the same as the dropdown.
    fireEvent.click(tab('Shipping'))
    expect($boardSlug.get()).toBe('')
    await chooseFromMenu('Unpin board')
    expect(tabNames()).toEqual(['Research'])

    view.unmount()
    disposeApi()
    bindPersistence()
    mount()
    expect((await screen.findByRole('group', { name: 'Pinned boards' })).textContent).toBe('Research')
  })

  it('keeps pins per connection and drops a drag that crossed a connection switch', async () => {
    const storage = bindPersistence()
    const remote = () => act(() => setConnection({ connectionId: 'remote-test', mode: 'remote' } as never))
    const local = () => act(() => setConnection({ mode: 'local' } as never))

    const pinBoth = async () => {
      await chooseFromMenu('Pin board')
      await chooseFromMenu(/^Research/)
      await chooseFromMenu('Pin board')
    }

    mount()
    await pinBoth()
    remote()
    expect(screen.queryByRole('group', { name: 'Pinned boards' })).toBeNull()
    await pinBoth()
    expect(storage.get('pinnedBoards.remote-test', [])).toEqual(['shipping', 'research'])

    // Both board lists are cached now, so the tabs stay mounted across the
    // switch and the drag's start survives it.
    local()
    expect(tabNames()).toEqual(['Shipping', 'Research'])
    const dataTransfer = { dropEffect: '', effectAllowed: '' }
    fireEvent.dragStart(tab('Research'), { dataTransfer })
    remote()
    fireEvent.dragOver(tab('Shipping'), { dataTransfer })
    fireEvent.drop(tab('Shipping'), { dataTransfer })
    expect(tabNames()).toEqual(['Shipping', 'Research'])
    expect(storage.get('pinnedBoards.remote-test', [])).toEqual(['shipping', 'research'])
    expect(storage.get('pinnedBoards', [])).toEqual(['shipping', 'research'])
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
