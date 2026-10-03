import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as ApiModule from './api'
import { AchievementsPage } from './page'
import type { Achievement, AchievementsResponse } from './types'

const mocks = vi.hoisted(() => ({
  fetchAchievements: vi.fn(),
  rescanAchievements: vi.fn()
}))

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof ApiModule>()),
  fetchAchievements: mocks.fetchAchievements,
  rescanAchievements: mocks.rescanAchievements
}))

const achievement = (overrides: Partial<Achievement> = {}): Achievement => ({
  category: 'Tool Mastery',
  criteria: 'Requirement: lifetime terminal calls. Tier ladder: Copper 10.',
  description: 'Spend serious time in shell-land.',
  discovered: true,
  icon: 'terminal',
  id: 'terminal_goblin',
  kind: 'lifetime',
  name: 'Terminal Goblin',
  next_threshold: 10,
  next_tier: 'Copper',
  progress: 4,
  progress_pct: 40,
  state: 'discovered',
  tier: null,
  unlocked: false,
  ...overrides
})

const response = (overrides: Partial<AchievementsResponse> = {}): AchievementsResponse => ({
  achievements: [achievement()],
  discovered_count: 1,
  error: null,
  generated_at: 1,
  is_stale: false,
  scan_meta: { mode: 'incremental', status: { state: 'idle' } },
  secret_count: 0,
  total_count: 1,
  unlocked_count: 0,
  ...overrides
})

function mount() {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <AchievementsPage />
    </QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('achievements page states', () => {
  it('renders the accessible loading state', () => {
    mocks.fetchAchievements.mockReturnValue(new Promise(() => undefined))

    mount()

    expect(screen.getByRole('status', { name: 'Reading achievements' })).toBeTruthy()
  })

  it('renders a request error with retry', async () => {
    mocks.fetchAchievements.mockRejectedValue(new Error('backend offline'))

    mount()

    expect(await screen.findByText('Achievements unavailable')).toBeTruthy()
    expect(screen.getByText('backend offline')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
  })

  it('renders a fresh pending catalog as a scan state instead of an empty state', async () => {
    mocks.fetchAchievements.mockResolvedValue(
      response({
        achievements: [achievement({ discovered: false, icon: 'secret', name: '???', state: 'secret' })],
        is_stale: true,
        scan_meta: { mode: 'pending', status: { state: 'running' } },
        secret_count: 1
      })
    )

    mount()

    expect(await screen.findByText('Preparing the first scan of your session history.')).toBeTruthy()
    expect(screen.getByText('???')).toBeTruthy()
    expect(screen.queryByText('No achievements yet')).toBeNull()
  })

  it('surfaces stale progress and nested scan failure while keeping backend achievements visible', async () => {
    mocks.fetchAchievements.mockResolvedValue(
      response({
        is_stale: true,
        scan_meta: { mode: 'failed', status: { last_error: 'could not read sessions', state: 'failed' } }
      })
    )

    mount()

    expect((await screen.findByRole('alert')).textContent).toContain('The latest scan failed')
    expect(screen.getByRole('alert').textContent).toContain('could not read sessions')
    expect(screen.getByText('Tool Mastery')).toBeTruthy()
    expect(screen.getByText('Terminal Goblin')).toBeTruthy()
  })

  it('renders scan progress and disables manual rescan while a scan is active', async () => {
    mocks.fetchAchievements.mockResolvedValue(
      response({
        scan_meta: {
          mode: 'in_progress',
          sessions_expected_total: 100,
          sessions_scanned_so_far: 25,
          status: { state: 'running' }
        }
      })
    )

    mount()

    expect(await screen.findByText('25 of 100 sessions scanned')).toBeTruthy()
    expect(screen.getByRole('progressbar', { name: 'Scanning session history' }).getAttribute('aria-valuenow')).toBe('25')
    expect((screen.getByRole('button', { name: 'Scanning…' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('keeps secret progress hidden until the backend reveals it', async () => {
    mocks.fetchAchievements.mockResolvedValue(
      response({
        achievements: [achievement({ icon: 'secret', name: '???', state: 'secret' })],
        secret_count: 1
      })
    )

    mount()

    await screen.findByText('???')
    expect(screen.getByText('Hidden')).toBeTruthy()
    expect(screen.queryByText('4 / 10')).toBeNull()
  })

  it('filters backend-provided categories and achievements', async () => {
    mocks.fetchAchievements.mockResolvedValue(
      response({
        achievements: [
          achievement(),
          achievement({ category: 'Research/Web', id: 'citation_goblin', name: 'Citation Goblin' })
        ],
        total_count: 2
      })
    )

    mount()
    await screen.findByText('Terminal Goblin')

    fireEvent.change(screen.getByRole('textbox', { name: 'Filter achievements' }), { target: { value: 'citation' } })

    expect(screen.queryByText('Terminal Goblin')).toBeNull()
    expect(screen.getByText('Citation Goblin')).toBeTruthy()
    expect(screen.getByText('Research/Web')).toBeTruthy()
  })

  it('runs a manual rescan through the API action', async () => {
    const initial = response()
    const rescanned = response({ unlocked_count: 1 })

    mocks.fetchAchievements.mockResolvedValue(initial)
    mocks.rescanAchievements.mockResolvedValue({ ...rescanned, ok: true })

    mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Rescan' }))

    await waitFor(() => expect(mocks.rescanAchievements).toHaveBeenCalledTimes(1))
  })
})
