import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { renameProfile } from '@/hermes'
import { retireLocalProfileGateways } from '@/store/gateway'
import { renameProfileInRailPrefs } from '@/store/profile'
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

vi.mock('@/store/profile', () => ({
  renameProfileInRailPrefs: vi.fn()
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
  expect(order).toEqual(['retire', 'rename'])
  // The sessions moved with the directory: tabs / cached tails / remembered ids keyed by the
  // old name follow, else every restored tab 404s against a backend that no longer exists (#111868).
  expect(migrateTilesForProfile).toHaveBeenCalledWith('selena', 'renamed')
  // The rail order is what ⌘N resolves against: a name left behind there drops the profile
  // into the alphabetical tail and hands its slot to somebody else (#130397).
  expect(renameProfileInRailPrefs).toHaveBeenCalledWith('selena', 'renamed')
})

it('migrates the rail prefs for a remote rename, which moves no local session state', async () => {
  render(
    <RenameProfileDialog
      currentName="selena"
      onClose={vi.fn()}
      open
      scope={{ connectionId: 'remote-1', profile: 'selena' }}
    />
  )

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() =>
    expect(renameProfile).toHaveBeenCalledWith('selena', 'renamed', { connectionId: 'remote-1', profile: 'selena' })
  )
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
  // The rail prefs are desktop-local, so they are keyed by name on this machine too.
  expect(renameProfileInRailPrefs).toHaveBeenCalledWith('selena', 'renamed')
})

it('leaves the rail prefs alone when the default profile only changes its display name', async () => {
  render(<RenameProfileDialog currentName="default" isDefault onClose={vi.fn()} open />)

  fireEvent.change(screen.getByLabelText(/display name/i), { target: { value: 'Work' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(renameProfile).toHaveBeenCalledWith('default', 'Work'))
  // The id stays "default" — only the label moved, so there is no name to re-key.
  expect(renameProfileInRailPrefs).not.toHaveBeenCalled()
})

it('does not retire gateways when validation rejects the submit', async () => {
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} open />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(screen.getByText('Name is required.')).toBeTruthy())
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(renameProfile).not.toHaveBeenCalled()
})
