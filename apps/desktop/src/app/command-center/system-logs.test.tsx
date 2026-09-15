import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getLogs } from '@/hermes'

import { CommandCenterView } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: vi.fn(),
  getLogs: vi.fn(),
  getStatus: vi.fn(() => Promise.resolve({ version: '0.21.0', gateway_running: true, active_sessions: 0 })),
  getUsageAnalytics: vi.fn(),
  restartGateway: vi.fn(),
  updateHermes: vi.fn()
}))
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn() }))
vi.mock('./maintenance', () => ({ MaintenancePanel: () => null }))

function renderSystem() {
  return render(
    <MemoryRouter>
      <CommandCenterView
        initialSection="system"
        onClose={() => {}}
        onDeleteSession={vi.fn()}
        onOpenSession={() => {}}
      />
    </MemoryRouter>
  )
}

function choose(group: string, label: string) {
  fireEvent.click(
    within(screen.getByRole('group', { name: group })).getAllByRole('button', { name: label })[0]
  )
}

describe('Command Center log filters', () => {
  beforeEach(() => {
    vi.mocked(getLogs).mockReset()
    vi.mocked(getLogs).mockResolvedValue({ file: 'agent', lines: ['WARNING example entry'] })
  })
  afterEach(cleanup)

  it('distinguishes the log file from severity and All levels keeps the selected file', async () => {
    renderSystem()
    await screen.findByText('WARNING example entry')
    expect(getLogs).toHaveBeenLastCalledWith({ file: 'agent', level: 'WARNING', lines: 100 })

    choose('Log file', 'errors.log')
    await waitFor(() => expect(getLogs).toHaveBeenLastCalledWith({ file: 'errors', level: 'WARNING', lines: 100 }))
    choose('Level', 'All levels')
    await waitFor(() => expect(getLogs).toHaveBeenLastCalledWith({ file: 'errors', level: 'ALL', lines: 100 }))

    for (const file of ['gateway', 'desktop', 'agent']) {
      choose('Log file', `${file}.log`)
      await waitFor(() => expect(getLogs).toHaveBeenLastCalledWith({ file, level: 'ALL', lines: 100 }))
    }

    for (const level of ['INFO', 'WARNING', 'ERROR']) {
      choose('Level', level.toLowerCase())
      await waitFor(() => expect(getLogs).toHaveBeenLastCalledWith({ file: 'agent', level, lines: 100 }))
    }
  })

  it('shows a search-specific empty state and restores results when the query is cleared', async () => {
    renderSystem()
    await screen.findByText('WARNING example entry')
    const search = screen.getByPlaceholderText('Filter log lines...')
    fireEvent.change(search, { target: { value: 'no such entry' } })
    expect(screen.getByText('No log lines match the search.')).toBeTruthy()
    expect(screen.getByText('Showing up to 100 recent lines from the selected file and level.')).toBeTruthy()
    fireEvent.change(search, { target: { value: '' } })
    expect(screen.getByText('WARNING example entry')).toBeTruthy()
  })

  it('does not replace the newly selected file with a late response from the previous file', async () => {
    let resolveOld!: (value: Awaited<ReturnType<typeof getLogs>>) => void
    vi.mocked(getLogs).mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveOld = resolve
        })
    )
    vi.mocked(getLogs).mockResolvedValue({ file: 'gateway', lines: ['new gateway entry'] })
    renderSystem()
    await waitFor(() => expect(getLogs).toHaveBeenCalledTimes(1))
    choose('Log file', 'gateway.log')
    await screen.findByText('new gateway entry')
    await act(async () => resolveOld({ file: 'agent', lines: ['old agent entry'] }))
    expect(screen.queryByText('old agent entry')).toBeNull()
    expect(screen.getByText('new gateway entry')).toBeTruthy()
  })
})
