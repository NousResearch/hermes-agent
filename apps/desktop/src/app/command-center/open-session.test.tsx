import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { SessionInfo } from '@/hermes'
import { $sessions } from '@/store/session'

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

  it('passes the clicked ROW to onOpenSession, not a bare id', async () => {
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

    const title = await screen.findByText('Remote conversation')
    fireEvent.click(title.closest('button') as HTMLButtonElement)

    expect(onOpenSession).toHaveBeenCalledWith(SESSION.id, SESSION)
  })
})
