/**
 * The done column holds a board's whole history, which on a long-lived board is thousands of cards.
 * Painting every one of them on each board change made the page crawl, so a long column mounts only
 * the rows near the viewport.
 * Nothing is capped or hidden: every card stays reachable by scrolling. These mount the REAL page
 * with only the REST layer and the drawer mocked.
 *
 * jsdom has no layout, so the geometry the virtualizer reads (the scroller's height, a row's height)
 * is stubbed with fixed numbers below; the virtualizer's own windowing then runs for real.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug, boardKey } from './api'
import { KanbanBoardPage } from './board'
import { KANBAN_LOCALES } from './i18n'
import type { KanbanBoard, KanbanTask } from './types'

const DONE_COUNT = 2_000
const VIEWPORT_PX = 600
const ROW_PX = 100
// The column's own 0.5rem (gap-2) between cards.
const ROW_GAP_PX = 8

const doneTitle = (n: number) => `done card ${String(n).padStart(4, '0')}`

const card = (id: string, status: string, title: string, over: Partial<KanbanTask> = {}): KanbanTask => ({
  id,
  status,
  title,
  ...over
})

// Newest first, as the server lists the done column.
const done = (count = DONE_COUNT): KanbanTask[] =>
  Array.from({ length: count }, (_, n) => card(`t_done_${n}`, 'done', doneTitle(n)))

const bigBoard = (count = DONE_COUNT): KanbanBoard => ({
  assignees: [],
  columns: [
    { name: 'ready', tasks: [card('t_r1', 'ready', 'ready card 1'), card('t_r2', 'ready', 'ready card 2')] },
    { name: 'running', tasks: [card('t_run', 'running', 'running card')] },
    { name: 'done', tasks: done(count) }
  ],
  latest_event_id: 5,
  now: 1_800_000_000,
  tenants: []
})

let currentBoard: KanbanBoard = bigBoard()

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoard: vi.fn(async () => currentBoard),
  fetchBoards: vi.fn(async () => ({ boards: [{ name: 'Main', slug: 'main', total: 1 }], current: 'main' })),
  fetchOrchestration: vi.fn(async () => ({ default_assignee: '' })),
  fetchProfiles: vi.fn(async () => ({ profiles: [] }))
}))

// The drawer has its own tests; here it only shows which card the board asked it to open.
vi.mock('./drawer', () => ({
  TaskDrawer: ({ id }: { id: null | string }) => (id ? <div data-testid="drawer">{id}</div> : null)
}))

const proto = HTMLElement.prototype
const realHeight = Object.getOwnPropertyDescriptor(proto, 'offsetHeight')
const realWidth = Object.getOwnPropertyDescriptor(proto, 'offsetWidth')

let disposeLocales: () => void = () => undefined
let client = new QueryClient()

beforeEach(() => {
  currentBoard = bigBoard()
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)

  Object.defineProperty(proto, 'offsetHeight', {
    configurable: true,
    get(this: HTMLElement) {
      if (this.dataset.kanbanScroller !== undefined) {
        return VIEWPORT_PX
      }

      return this.dataset.index !== undefined ? ROW_PX : 0
    }
  })

  Object.defineProperty(proto, 'offsetWidth', {
    configurable: true,
    get(this: HTMLElement) {
      return this.dataset.kanbanScroller !== undefined || this.dataset.index !== undefined ? 256 : 0
    }
  })
})

afterEach(() => {
  cleanup()
  disposeLocales()
  $boardSlug.set('')

  if (realHeight) {
    Object.defineProperty(proto, 'offsetHeight', realHeight)
  }

  if (realWidth) {
    Object.defineProperty(proto, 'offsetWidth', realWidth)
  }
})

const withQuery = (ui: ReactNode) => <QueryClientProvider client={client}>{ui}</QueryClientProvider>

const mountedDone = () => screen.queryAllByText(/^done card \d{4}$/)
// eslint-disable-next-line no-restricted-globals -- a test reads the jsdom document it rendered into
const scrollerOf = (name: string) => document.querySelector<HTMLElement>(`[data-kanban-scroller="${name}"]`)!

const scrollTo = async (scroller: HTMLElement, top: number) => {
  Object.defineProperty(scroller, 'scrollTop', { configurable: true, value: top, writable: true })
  await act(async () => {
    fireEvent.scroll(scroller)
  })
}

const sizerHeight = (scroller: HTMLElement) =>
  parseFloat(scroller.querySelector<HTMLElement>('[data-kanban-sizer]')!.style.height)

/** Where a mounted row sits in the scroll area, read from the transform the virtualizer put on it. */
const rowStart = (title: string) => {
  const row = screen.getByText(title).closest<HTMLElement>('[data-index]')!

  return parseFloat(/translateY\(([\d.]+)px\)/.exec(row.style.transform)![1])
}

/** Scroll until card `n` of the done column is mounted: a real user drags, and rows they have not
 *  seen yet are laid out at the estimate until measured, so aim from a row that IS mounted. */
const scrollToCard = async (scroller: HTMLElement, n: number) => {
  for (let attempt = 0; attempt < 12 && !screen.queryByText(doneTitle(n)); attempt += 1) {
    const rows = [...scroller.querySelectorAll<HTMLElement>('[data-index]')]
    const first = rows[0]
    const firstIndex = Number(first.dataset.index)
    const firstStart = parseFloat(/translateY\(([\d.]+)px\)/.exec(first.style.transform)![1])

    await scrollTo(scroller, Math.max(0, firstStart + (n - firstIndex) * (ROW_PX + ROW_GAP_PX)))
  }

  await waitFor(() => expect(screen.queryByText(doneTitle(n))).not.toBeNull())
}

async function mountBoard() {
  render(withQuery(<KanbanBoardPage />))
  await screen.findByText(doneTitle(0))
}

describe('the done column paints a window, not the history', () => {
  it('mounts far fewer than 2,000 cards, and the scroller still has room for every one', async () => {
    await mountBoard()

    expect(mountedDone().length).toBeGreaterThan(0)
    expect(mountedDone().length).toBeLessThan(100)

    // Every card has its slot: jsdom has no layout, so the browser's scrollHeight is read from the sizer
    // whose inline height is exactly what the scroll area is made of.
    expect(sizerHeight(scrollerOf('done'))).toBeGreaterThanOrEqual(DONE_COUNT * ROW_PX)
  })

  it('the count in the header and the board total still say 2,000 (nothing is hidden)', async () => {
    await mountBoard()

    expect(screen.getByText(String(DONE_COUNT))).toBeTruthy()
    expect(screen.getByText(String(DONE_COUNT + 3))).toBeTruthy()
  })

  it('scrolling to the bottom mounts the oldest card and unmounts the newest', async () => {
    await mountBoard()

    const scroller = scrollerOf('done')

    await scrollTo(scroller, sizerHeight(scroller) - VIEWPORT_PX)

    await waitFor(() => expect(screen.queryByText(doneTitle(DONE_COUNT - 1))).not.toBeNull())
    expect(screen.queryByText(doneTitle(0))).toBeNull()
    expect(mountedDone().length).toBeLessThan(100)
  })

  it('leaves a short column as a plain list: every card mounted, no sizer', async () => {
    await mountBoard()

    expect(screen.getByText('ready card 1')).toBeTruthy()
    expect(screen.getByText('ready card 2')).toBeTruthy()
    expect(scrollerOf('ready').querySelector('[data-kanban-sizer]')).toBeNull()
  })

  it('keeps the add-card button after the last row of the scroller', async () => {
    await mountBoard()

    const scroller = scrollerOf('done')
    const children = [...scroller.children]
    const sizerIndex = children.findIndex(child => child.hasAttribute('data-kanban-sizer'))
    const addIndex = children.findIndex(child => child.tagName === 'BUTTON')

    expect(sizerIndex).toBeGreaterThanOrEqual(0)
    expect(addIndex).toBeGreaterThan(sizerIndex)
  })
})

describe('the virtual column keeps the board interactions', () => {
  it('a click on a mounted card opens it in the drawer', async () => {
    await mountBoard()

    fireEvent.click(screen.getByText(doneTitle(2)))

    expect(screen.getByTestId('drawer').textContent).toBe('t_done_2')
  })

  it('a modifier-click selects a card, and the selection survives the card being scrolled away and back', async () => {
    await mountBoard()

    fireEvent.click(screen.getByText(doneTitle(1)), { ctrlKey: true })
    expect(screen.getByText('1 selected')).toBeTruthy()

    const scroller = scrollerOf('done')

    await scrollTo(scroller, sizerHeight(scroller) - VIEWPORT_PX)
    await waitFor(() => expect(screen.queryByText(doneTitle(1))).toBeNull())
    expect(screen.getByText('1 selected')).toBeTruthy()

    await scrollTo(scroller, 0)
    await waitFor(() => expect(screen.queryByText(doneTitle(1))).not.toBeNull())

    // The remounted card paints as selected again (the page owns the set, not the row).
    const cardEl = screen.getByText(doneTitle(1)).closest('[draggable]')!

    expect(cardEl.className).toContain('border-(--dt-composer-ring)')
    expect(screen.getByText('1 selected')).toBeTruthy()
  })

  it('a client-side filter narrows the column and the window follows', async () => {
    await mountBoard()

    fireEvent.change(screen.getByLabelText('Filter cards…'), { target: { value: 'done card 1999' } })

    await waitFor(() => expect(mountedDone()).toHaveLength(1))
    expect(screen.getByText(doneTitle(DONE_COUNT - 1))).toBeTruthy()
  })
})

describe('a board change while the user is reading history', () => {
  it('keeps the card on screen where it was when a new card lands at the top', async () => {
    await mountBoard()

    const scroller = scrollerOf('done')

    await scrollToCard(scroller, 500)

    const before = rowStart(doneTitle(500)) - scroller.scrollTop
    const top = scroller.scrollTop

    expect(top).toBeGreaterThan(0)

    await act(async () => {
      client.setQueryData(boardKey('local', '', false), {
        ...currentBoard,
        columns: currentBoard.columns.map(col =>
          col.name === 'done'
            ? { ...col, tasks: [card('t_done_new', 'done', 'done card just now'), ...col.tasks] }
            : col
        )
      })
    })

    // The new row sits above the reader, so the scroll offset moved with it.
    await waitFor(() => expect(scroller.scrollTop).toBeGreaterThan(top))
    expect(rowStart(doneTitle(500)) - scroller.scrollTop).toBeCloseTo(before, 5)
  })

  it('does not move the reader when nothing above them changed', async () => {
    await mountBoard()

    const scroller = scrollerOf('done')

    await scrollToCard(scroller, 500)

    const top = scroller.scrollTop

    await act(async () => {
      client.setQueryData(boardKey('local', '', false), { ...currentBoard, now: currentBoard.now + 60 })
    })

    expect(scroller.scrollTop).toBe(top)
  })
})
