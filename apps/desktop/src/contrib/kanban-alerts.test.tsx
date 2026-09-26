import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import type { ReadableAtom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { ROUTES_AREA, SIDEBAR_NAV_AREA, type SidebarNavContribution } from '@/app/routes'
import { createPluginContext } from '@/contrib/plugin'
import { registry } from '@/contrib/registry'

import { $boardSlug } from '../plugins/kanban/api'
import { $unseenByBoard, cursorKey } from '../plugins/kanban/completion-notify'
import plugin from '../plugins/kanban/plugin'

// #123596 wiring through the real registration: the Kanban nav row carries the
// unseen count, and mounting the board page clears that board's count.

vi.mock('../plugins/kanban/api', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  bindApi: () => () => {}
}))

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $unseenByBoard.set({})
  $boardSlug.set('')
})

const register = () => plugin.register(createPluginContext('kanban', dispose => disposers.push(dispose)))

it('the Kanban nav row counts unseen events on the active connection', () => {
  register()
  const nav = registry.getArea(SIDEBAR_NAV_AREA).find(c => c.id === 'kanban:nav')!.data as SidebarNavContribution
  const count = nav.count as ReadableAtom<number>

  expect(count.get()).toBe(0)
  $unseenByBoard.set({ [cursorKey('local', 'ops')]: 2, [cursorKey('local', 'research')]: 1 })
  expect(count.get()).toBe(3)
  expect(nav.countLabel?.(3)).toBe('3 unseen task updates')

  $unseenByBoard.set({})
  expect(count.get()).toBe(0)
})

it('opening the board page clears that board, not the others', async () => {
  register()
  $boardSlug.set('ops')
  $unseenByBoard.set({ [cursorKey('local', 'ops')]: 2, [cursorKey('local', 'research')]: 1 })

  const page = registry.getArea(ROUTES_AREA).find(c => c.id === 'kanban:page')!
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  await act(async () => {
    render(<QueryClientProvider client={client}>{page.render!()}</QueryClientProvider>)
  })

  expect($unseenByBoard.get()).toEqual({ [cursorKey('local', 'research')]: 1 })

  // Switching boards inside the mounted page (the in-page switcher sets the
  // slug; the route does not remount) clears the newly opened board too.
  act(() => $boardSlug.set('research'))
  expect($unseenByBoard.get()).toEqual({})
})
