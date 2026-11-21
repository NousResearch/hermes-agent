import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { SessionInfo } from '@/hermes'
import { $sessions } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'

import { CommandCenterView } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: vi.fn(() => Promise.resolve({ running: false })),
  getLogs: vi.fn(() => Promise.resolve({ lines: [] })),
  getStatus: vi.fn(() => Promise.resolve({})),
  getUsageAnalytics: vi.fn(() => Promise.resolve({})),
  restartGateway: vi.fn(),
  updateHermes: vi.fn()
}))
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn() }))
vi.mock('./maintenance', () => ({ MaintenancePanel: () => null }))

afterEach(() => {
  cleanup()
  $sessions.set([])
  $shellChoice.set(null)
})

const SESSION = {
  connection_id: 'gw-tailscale',
  ended_at: null,
  id: 'sess-remote',
  input_tokens: 0,
  is_active: false,
  last_active: 1_756_600_000,
  message_count: 3,
  model: null,
  output_tokens: 0,
  profile: 'research',
  started_at: 1_756_500_000,
  title: 'Remote conversation'
} as SessionInfo

// Same open-path contract as the cron run rows (#82527): a list row hands its
// owner (connection, profile) to the open call so a remote/other-profile row
// resumes on the backend that owns it, not the ambient one.
describe('Command Center session open carries the row', () => {
  beforeEach(() => {
    $sessions.set([SESSION])
  })

  it.each([false, true])('passes the clicked ROW to onOpenSession in simplified mode %s', async simplified => {
    $shellChoice.set(simplified)
    $sessions.set([SESSION, { ...SESSION, id: 'newer', title: 'Recent notes', last_active: SESSION.last_active! + 100 }])
    const onOpenSession = vi.fn()
    render(
      <MemoryRouter>
        <CommandCenterView
          initialSection="sessions"
          onClose={() => {}}
          onDeleteSession={() => Promise.resolve()}
          onOpenSession={onOpenSession}
        />
      </MemoryRouter>
    )

    expect((await screen.findAllByText(/^(Remote conversation|Recent notes)$/)).map(node => node.textContent)).toEqual([
      'Recent notes', 'Remote conversation'
    ])
    fireEvent.change(screen.getByPlaceholderText('Search sessions, views, and actions'), { target: { value: 'Remote' } })
    await waitFor(() => expect(screen.queryByText('Recent notes')).toBeNull())
    const title = screen.getByText('Remote conversation')
    fireEvent.click(title.closest('button') as HTMLButtonElement)

    expect(onOpenSession).toHaveBeenCalledWith(SESSION.id, SESSION)
  })
})
