/**
 * Board pause UI (#141451): header toggle, warning banner, red nav badge.
 *
 * The contract the tests pin: the banner explains what pausing DOES (nothing
 * new starts, running workers finish), the toggle calls the right endpoint for
 * the current state, and the nav badge — the "visible from anywhere" surface —
 * only shows while the board payload reports a pause.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug, fetchBoards } from './api'
import { KANBAN_LOCALES } from './i18n'
import { KanbanPauseNavBadge, PauseBanner, PauseButton } from './pause'
import type { KanbanBoard } from './types'

const pauseBoard = vi.fn(async () => ({ board: 'default', pause: { board: 'default', paused_at: 'x', reason: null } }))
const resumeBoard = vi.fn(async () => ({ board: 'default', pause: null }))

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoard: vi.fn(),
  fetchBoards: vi.fn(),
  pauseBoard: (...args: unknown[]) => pauseBoard(...(args as [])),
  resumeBoard: (...args: unknown[]) => resumeBoard(...(args as []))
}))

const boardsWith = (paused: string[]) => ({
  boards: [
    { name: 'Default', slug: 'default', pause: paused.includes('default') ? { board: 'default', paused_at: 'x', reason: 'reloading the provider' } : null },
    { name: 'Sumaris', slug: 'sumaris', pause: paused.includes('sumaris') ? { board: 'sumaris', paused_at: 'x', reason: null } : null }
  ],
  current: 'default'
})

const emptyBoard: KanbanBoard = {
  assignees: [],
  columns: [],
  latest_event_id: 0,
  now: 0,
  pause: null,
  tenants: []
}

const pausedBoard: KanbanBoard = {
  ...emptyBoard,
  pause: { board: 'default', paused_at: '2026-10-20T12:00:00+00:00', reason: 'reloading the provider' }
}

let disposeLocales: () => void = () => undefined

beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  vi.clearAllMocks()
  $boardSlug.set('')
})

const mount = (node: React.ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {node}
    </QueryClientProvider>
  )

describe('pause banner', () => {
  it('says what a pause means and echoes the operator reason', () => {
    mount(<PauseBanner pause={pausedBoard.pause!} />)

    expect(screen.getByText('Board paused')).toBeTruthy()
    expect(screen.getByText(/No new task will start/)).toBeTruthy()
    expect(screen.getByText('reloading the provider')).toBeTruthy()
  })

  it('resumes from the banner', async () => {
    mount(<PauseBanner pause={pausedBoard.pause!} />)
    fireEvent.click(screen.getByRole('button', { name: /Resume board/ }))
    await vi.waitFor(() => expect(resumeBoard).toHaveBeenCalledTimes(1))
  })
})

describe('pause toggle', () => {
  it('offers Pause while running and posts /pause', async () => {
    mount(<PauseButton paused={false} />)
    const btn = screen.getByRole('button', { name: 'Pause board' })
    // Outline, not a filled/primary action: pausing is a quiet toggle next to
    // "New task", the way the cron panel's pause reads.
    expect(btn.className).not.toContain('bg-primary')
    expect(btn.className).toContain('shadow-[inset_0_0_0_1px')
    fireEvent.click(btn)
    await vi.waitFor(() => expect(pauseBoard).toHaveBeenCalledTimes(1))
    expect(resumeBoard).not.toHaveBeenCalled()
  })

  it('offers Resume while paused and posts /resume', async () => {
    mount(<PauseButton paused />)
    fireEvent.click(screen.getByRole('button', { name: 'Resume board' }))
    await vi.waitFor(() => expect(resumeBoard).toHaveBeenCalledTimes(1))
    expect(pauseBoard).not.toHaveBeenCalled()
  })
})

describe('sidebar badge', () => {
  it('names a held board that is NOT the one on screen', async () => {
    vi.mocked(fetchBoards).mockResolvedValue(boardsWith(['sumaris']))
    mount(<KanbanPauseNavBadge />)

    const badge = await screen.findByTestId('kanban-nav-paused')
    expect(badge.textContent).toBe('PAUSED · Sumaris')
    expect(badge.getAttribute('title')).toBe('Sumaris')
  })

  it('drops the name when the board on screen is the held one', async () => {
    vi.mocked(fetchBoards).mockResolvedValue(boardsWith(['default']))
    mount(<KanbanPauseNavBadge />)

    const badge = await screen.findByTestId('kanban-nav-paused')
    expect(badge.textContent).toBe('PAUSED')
    expect(badge.getAttribute('title')).toBe('Default — reloading the provider')
  })

  it('counts several held boards instead of lying about one', async () => {
    vi.mocked(fetchBoards).mockResolvedValue(boardsWith(['default', 'sumaris']))
    mount(<KanbanPauseNavBadge />)

    const badge = await screen.findByTestId('kanban-nav-paused')
    expect(badge.textContent).toContain('2 PAUSED')
    expect(badge.getAttribute('title')).toBe('Boards on pause: Default, Sumaris')
  })

  it('renders nothing while every board runs', async () => {
    vi.mocked(fetchBoards).mockResolvedValue(boardsWith([]))
    const { container } = mount(<KanbanPauseNavBadge />)

    await vi.waitFor(() => expect(fetchBoards).toHaveBeenCalled())
    expect(container.querySelector('[data-testid="kanban-nav-paused"]')).toBeNull()
  })
})
