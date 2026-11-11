import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { renameProfile } from '@/hermes'
import { $hiddenSidebarAgents, agentSidebarKey, setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'
import { retireLocalProfileGateways } from '@/store/gateway'
import { migrateTilesForProfile } from '@/store/session-states'

import { RenameProfileDialog } from './rename-profile-dialog'

// Pins the rename half of the deleted-profile-resurrection class (#88638 fixed
// the delete half): a retained renderer socket for the OLD profile name must be
// retired BEFORE the rename PATCH tears down its backend, or the socket's
// reconnect loop respawns the old-name backend and recreates the directory the
// rename just moved.

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $hiddenSidebarAgents.set([])
})

vi.mock('@/hermes', () => ({
  renameProfile: vi.fn(async () => ({ name: 'renamed', ok: true, path: '/x' }))
}))

vi.mock('@/store/gateway', () => ({
  retireLocalProfileGateways: vi.fn()
}))

vi.mock('@/store/session-states', () => ({
  migrateTilesForProfile: vi.fn()
}))

it.each([undefined, { connectionId: 'local', profile: 'selena' }])(
  'retires old local gateways before renaming and migrates ownership (scope: %j)',
  async scope => {
    const order: string[] = []

    vi.mocked(retireLocalProfileGateways).mockImplementationOnce(() => {
      order.push('retire')
    })
    vi.mocked(renameProfile).mockImplementationOnce(async () => {
      order.push('rename')

      return { name: 'renamed', ok: true, path: '/x' }
    })

    render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} open scope={scope} />)

    fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
    fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

    await waitFor(() => expect(renameProfile).toHaveBeenCalledWith('selena', 'renamed', ...(scope ? [scope] : [])))
    expect(retireLocalProfileGateways).toHaveBeenCalledWith('selena')
    expect(order).toEqual(['retire', 'rename'])
    expect(migrateTilesForProfile).toHaveBeenCalledWith('selena', 'renamed')
  }
)

it('does not retire gateways when validation rejects the submit', async () => {
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} open />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(screen.getByText('Name is required.')).toBeTruthy())
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(renameProfile).not.toHaveBeenCalled()
})

it('moves a remote hidden preference without changing the same-named local agent', async () => {
  const remote = { connectionId: 'remote', profile: 'selena' }
  const local = { connectionId: null, profile: 'selena' }
  setSidebarAgentHidden(remote, true)
  setSidebarAgentHidden(local, true)
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} open scope={remote} />)
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))
  await waitFor(() =>
    expect($hiddenSidebarAgents.get()).toEqual([
      agentSidebarKey({ ...remote, profile: 'renamed' }),
      agentSidebarKey(local)
    ])
  )
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
})
