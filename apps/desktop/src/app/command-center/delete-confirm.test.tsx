import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { SessionInfo } from '@/hermes'
import { $sessions } from '@/store/session'

import { CommandCenterView } from './index'

// #99410: the Command Center → Sessions trash button hard-deleted the session
// (row + messages + request_dump files) instantly, with no confirmation —
// e6708af1f (#61470) guarded the sidebar/tab/header entry points via
// DeleteSessionDialog but missed this independent one. The delete must be
// gated behind the shared ConfirmDialog: no onDeleteSession call on the trash
// click alone, the call only after an explicit confirm, and never on cancel.

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

afterEach(cleanup)

const SESSION: SessionInfo = {
  ended_at: null,
  id: 'sess-1',
  input_tokens: 0,
  is_active: false,
  last_active: 1_756_600_000,
  message_count: 3,
  model: null,
  output_tokens: 0,
  started_at: 1_756_500_000,
  title: 'Precious conversation'
} as SessionInfo

function renderCommandCenter(onDeleteSession: (session: SessionInfo) => Promise<void>) {
  return render(
    <MemoryRouter>
      <CommandCenterView
        initialSection="sessions"
        onClose={() => {}}
        onDeleteSession={onDeleteSession}
        onOpenSession={() => {}}
      />
    </MemoryRouter>
  )
}

describe('Command Center session delete confirmation (#99410)', () => {
  beforeEach(() => {
    $sessions.set([SESSION])
  })

  it('does NOT delete on trash click alone — a confirm dialog appears instead', async () => {
    const onDeleteSession = vi.fn(() => Promise.resolve())
    renderCommandCenter(onDeleteSession)

    fireEvent.click(await screen.findByRole('button', { name: 'Delete session' }))

    expect(onDeleteSession).not.toHaveBeenCalled()
    expect(await screen.findByRole('dialog')).toBeTruthy()
  })

  it('deletes only after explicit confirm', async () => {
    const onDeleteSession = vi.fn(() => Promise.resolve())
    renderCommandCenter(onDeleteSession)

    fireEvent.click(await screen.findByRole('button', { name: 'Delete session' }))
    const dialog = await screen.findByRole('dialog')
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(onDeleteSession).toHaveBeenCalledWith(SESSION))
    expect(dialog).toBeTruthy()
  })

  it('cancel closes the dialog without deleting', async () => {
    const onDeleteSession = vi.fn(() => Promise.resolve())
    renderCommandCenter(onDeleteSession)

    fireEvent.click(await screen.findByRole('button', { name: 'Delete session' }))
    await screen.findByRole('dialog')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(onDeleteSession).not.toHaveBeenCalled()
  })
})
