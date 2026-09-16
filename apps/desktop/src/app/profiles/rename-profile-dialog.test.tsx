import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { renameProfile } from '@/hermes'
import { retireLocalProfileGateways } from '@/store/gateway'
import { stageProfileRenameState } from '@/store/profile-rename-state'

import { RenameProfileDialog } from './rename-profile-dialog'

// Pins the rename half of the deleted-profile-resurrection class (#88638 fixed
// the delete half): a retained renderer socket for the OLD profile name must be
// retired BEFORE the rename PATCH tears down its backend, or the socket's
// reconnect loop respawns the old-name backend and recreates the directory the
// rename just moved.

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

vi.mock('@/hermes', () => ({
  renameProfile: vi.fn(async () => ({ name: 'renamed', ok: true, path: '/x' }))
}))

vi.mock('@/store/gateway', () => ({
  retireLocalProfileGateways: vi.fn()
}))

vi.mock('@/store/profile-rename-state', () => ({
  cancelProfileRenameState: vi.fn(),
  completeProfileRenameState: vi.fn(),
  stageProfileRenameState: vi.fn()
}))

vi.mock('@/store/session', () => ({
  $connection: {
    get: () => ({
      baseUrl: 'https://gateway-a.example',
      connectionId: 'gateway-a',
      mode: 'remote',
      profile: 'default'
    })
  }
}))

it('retires the old-name local gateways before issuing the rename', async () => {
  const order: string[] = []

  vi.mocked(retireLocalProfileGateways).mockImplementationOnce(() => {
    order.push('retire')
  })
  vi.mocked(renameProfile).mockImplementationOnce(async () => {
    order.push('rename')

    return { name: 'renamed', ok: true, path: '/x' }
  })

  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} open />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(renameProfile).toHaveBeenCalledWith('selena', 'renamed'))
  expect(retireLocalProfileGateways).toHaveBeenCalledWith('selena')
  expect(stageProfileRenameState).toHaveBeenCalledWith('selena', 'renamed', {
    connectionId: 'local',
    oldNavigationSuffix: '',
    newNavigationSuffix: ''
  })
  expect(order).toEqual(['retire', 'rename'])
})

it('does not retire gateways when validation rejects the submit', async () => {
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} open />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(screen.getByText('Name is required.')).toBeTruthy())
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(renameProfile).not.toHaveBeenCalled()
})

it('does not borrow active navigation scope when renaming an inactive profile', async () => {
  const scope = { connectionId: 'gateway-a', profile: 'work' }

  render(<RenameProfileDialog currentName="work" onClose={vi.fn()} open scope={scope} />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'personal' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(renameProfile).toHaveBeenCalledWith('work', 'personal', scope))
  expect(stageProfileRenameState).toHaveBeenCalledWith('work', 'personal', {
    connectionId: 'gateway-a',
    oldNavigationSuffix: null,
    newNavigationSuffix: null
  })
})
