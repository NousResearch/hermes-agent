import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Exercise the real namespaced localStorage door and connection store.
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
    boards: [{ name: 'Shipping', project_id: null, slug: 'shipping', total: 3 }],
    current: 'shipping'
  }))
}))

let disposeLocales: () => void = () => undefined
let disposeApi: () => void = () => undefined

beforeEach(() => {
  localStorage.clear()
  setConnection(null)
  vi.mocked(fetchBoards).mockResolvedValue({
    boards: [
      { name: 'Shipping', project_id: null, slug: 'shipping', total: 3 },
      { name: 'Research', project_id: null, slug: 'research', total: 1 }
    ],
    current: 'shipping'
  })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeApi()
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

async function pinFromMenu(name: string) {
  fireEvent.pointerDown(await screen.findByRole('button', { name: /^Board:/ }), { button: 0, ctrlKey: false })
  const pins = await screen.findByRole('menuitem', { name: 'Pin boards' })
  fireEvent.keyDown(pins, { key: 'ArrowRight' })
  const checkbox = await screen.findByRole('menuitemcheckbox', { name })
  fireEvent.click(checkbox)
  fireEvent.keyDown(checkbox, { key: 'Escape' })
}

const pinnedButtons = () => within(screen.getByRole('group', { name: 'Pinned boards' })).getAllByRole('button')

describe('board switcher', () => {
  // The rename and settings dialogs stay mounted while closed, so they render
  // with a null board on every pass. Reading the slug inside their mutation
  // callback used to crash the whole contribution, because the React Compiler
  // lifts a callback's property reads into its render-time dependency check.
  it('renders while its dialogs are closed', async () => {
    mount()

    expect(await screen.findByText('Shipping')).toBeTruthy()
  })

  it('pins and reorders without navigating, restores the order after reload, and keeps unpinned boards reachable', async () => {
    const storage = bindPersistence()
    const view = mount()
    await pinFromMenu('Shipping')
    await pinFromMenu('Research')
    expect($boardSlug.get()).toBe('')
    expect(pinnedButtons().map(button => button.textContent)).toEqual(['Shipping', 'Research'])

    fireEvent.keyDown(pinnedButtons()[1], { key: 'ArrowLeft', altKey: true })
    expect(pinnedButtons().map(button => button.textContent)).toEqual(['Research', 'Shipping'])
    expect($boardSlug.get()).toBe('')

    const dataTransfer = { effectAllowed: '', dropEffect: '', setData: vi.fn() }
    fireEvent.dragStart(pinnedButtons()[1], { dataTransfer })
    fireEvent.dragOver(pinnedButtons()[0], { dataTransfer })
    fireEvent.drop(pinnedButtons()[0], { dataTransfer })
    expect(pinnedButtons().map(button => button.textContent)).toEqual(['Shipping', 'Research'])
    expect(storage.get('pinnedBoards', [])).toEqual(['shipping', 'research'])
    expect($boardSlug.get()).toBe('')

    fireEvent.click(pinnedButtons()[1])
    expect($boardSlug.get()).toBe('research')
    expect(pinnedButtons()[1].getAttribute('aria-pressed')).toBe('true')
    await pinFromMenu('Research')
    expect(pinnedButtons().map(button => button.textContent)).toEqual(['Shipping'])
    expect($boardSlug.get()).toBe('research')

    view.unmount()
    disposeApi()
    bindPersistence()
    mount()
    expect((await screen.findByRole('group', { name: 'Pinned boards' })).textContent).toBe('Shipping')
    fireEvent.pointerDown(await screen.findByRole('button', { name: /^Board:/ }), { button: 0, ctrlKey: false })
    fireEvent.click(await screen.findByRole('menuitem', { name: /Research/ }))
    expect($boardSlug.get()).toBe('research')
  })

  it('isolates pins across a connection round trip and cannot drop an outgoing drag into the new connection', async () => {
    const storage = bindPersistence()
    mount()
    await pinFromMenu('Shipping')
    await pinFromMenu('Research')
    const dataTransfer = { effectAllowed: '', dropEffect: '', setData: vi.fn() }
    fireEvent.dragStart(pinnedButtons()[1], { dataTransfer })

    act(() => setConnection({ connectionId: 'remote-test', mode: 'remote' } as never))
    expect(screen.queryByRole('group', { name: 'Pinned boards' })).toBeNull()
    await pinFromMenu('Research')
    await pinFromMenu('Shipping')
    fireEvent.drop(pinnedButtons()[1], { dataTransfer })
    expect(pinnedButtons().map(button => button.textContent)).toEqual(['Research', 'Shipping'])
    expect(storage.get('pinnedBoards.remote-test', [])).toEqual(['research', 'shipping'])
    expect(storage.get('pinnedBoards', [])).toEqual(['shipping', 'research'])

    act(() => setConnection({ mode: 'local' } as never))
    await screen.findByRole('group', { name: 'Pinned boards' })
    fireEvent.drop(pinnedButtons()[0], { dataTransfer })
    expect(pinnedButtons().map(button => button.textContent)).toEqual(['Shipping', 'Research'])
    expect(storage.get('pinnedBoards', [])).toEqual(['shipping', 'research'])
  })
})
