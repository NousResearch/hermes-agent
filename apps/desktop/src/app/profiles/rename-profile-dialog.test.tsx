import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ProfileScope, setApiRequestConnection, setApiRequestLocalMode } from '@/api/client'
import { renameProfile } from '@/hermes'
import { retireLocalProfileGateways } from '@/store/gateway'
import { migrateTilesForProfile } from '@/store/session-states'

import { RenameProfileDialog } from './rename-profile-dialog'

// Pins the rename half of the deleted-profile-resurrection class (#88638 fixed
// the delete half): a retained renderer socket for the OLD profile name must be
// retired BEFORE the rename PATCH tears down its backend, or the socket's
// reconnect loop respawns the old-name backend and recreates the directory the
// rename just moved.

beforeEach(() => {
  setApiRequestConnection(null)
  setApiRequestLocalMode(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  setApiRequestConnection(null)
  setApiRequestLocalMode(false)
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

it.each<{ label: string; scope: ProfileScope }>([
  { label: 'ambient', scope: undefined },
  { label: 'legacy string', scope: 'selena' },
  { label: 'profile-only', scope: { profile: 'selena' } },
  { label: 'explicit local', scope: { connectionId: 'local', profile: 'selena' } },
  { label: 'normalized explicit local', scope: { connectionId: ' local ', profile: 'selena' } }
])('retires and migrates the $label profile around its owning rename', async ({ scope }) => {
  if (scope && typeof scope === 'object' && scope.connectionId?.trim() === 'local') {
    setApiRequestConnection('homelab')
  }

  const order: string[] = []

  const onRenamed = vi.fn(() => {
    order.push('notified')
  })

  vi.mocked(retireLocalProfileGateways).mockImplementationOnce(() => {
    order.push('retire')
  })
  vi.mocked(renameProfile).mockImplementationOnce(async () => {
    order.push('rename')
    expect(migrateTilesForProfile).not.toHaveBeenCalled()

    return { name: 'renamed', ok: true, path: '/x' }
  })
  vi.mocked(migrateTilesForProfile).mockImplementationOnce(() => {
    order.push('migrate')
  })

  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} onRenamed={onRenamed} open scope={scope} />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(onRenamed).toHaveBeenCalledWith('renamed'))
  expect(vi.mocked(renameProfile).mock.calls[0]).toEqual(
    scope === undefined
      ? ['selena', 'renamed']
      : [
          'selena',
          'renamed',
          typeof scope === 'object' && scope?.connectionId
            ? { ...scope, connectionId: scope.connectionId.trim() }
            : scope
        ]
  )
  expect(retireLocalProfileGateways).toHaveBeenCalledWith('selena')
  expect(order).toEqual(['retire', 'rename', 'migrate', 'notified'])
  // The sessions moved with the directory: tabs / cached tails / remembered ids keyed by the
  // old name follow, else every restored tab 404s against a backend that no longer exists (#111868).
  expect(migrateTilesForProfile).toHaveBeenCalledWith('selena', 'renamed')
})

it('renames a remote-owned profile without retiring or migrating its same-named local profile', async () => {
  const scope = { connectionId: 'homelab', profile: 'selena' }
  const onRenamed = vi.fn()

  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} onRenamed={onRenamed} open scope={scope} />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(onRenamed).toHaveBeenCalledWith('renamed'))
  expect(renameProfile).toHaveBeenCalledWith('selena', 'renamed', scope)
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
})

it.each<{ label: string; scope: ProfileScope }>([
  { label: 'undefined', scope: undefined },
  { label: 'null', scope: null },
  { label: 'legacy string', scope: 'selena' },
  { label: 'profile-only', scope: { profile: 'selena' } },
  { label: 'blank connection', scope: { connectionId: '  ', profile: 'selena' } }
])('leaves local state untouched for an ambient remote $label scope', async ({ scope }) => {
  setApiRequestConnection('homelab')
  const onRenamed = vi.fn()
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} onRenamed={onRenamed} open scope={scope} />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(onRenamed).toHaveBeenCalledWith('renamed'))
  expect(renameProfile).toHaveBeenCalledWith('selena', 'renamed', {
    connectionId: 'homelab',
    profile: typeof scope === 'string' ? scope : scope?.profile || undefined
  })
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
})

it('retains an existing local registry tag when retirement changes the active connection', async () => {
  setApiRequestConnection('local')
  vi.mocked(retireLocalProfileGateways).mockImplementationOnce(() => {
    setApiRequestConnection('homelab')
  })
  const onRenamed = vi.fn()
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} onRenamed={onRenamed} open />)
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))
  await waitFor(() => expect(onRenamed).toHaveBeenCalledWith('renamed'))
  expect(renameProfile).toHaveBeenCalledWith('selena', 'renamed', { connectionId: 'local', profile: undefined })
  expect(migrateTilesForProfile).toHaveBeenCalledWith('selena', 'renamed')
})

it('does not assume an unknown legacy owner is local', async () => {
  setApiRequestLocalMode(false)
  const onRenamed = vi.fn()
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} onRenamed={onRenamed} open />)
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))
  await waitFor(() => expect(onRenamed).toHaveBeenCalledWith('renamed'))
  expect(renameProfile).toHaveBeenCalledWith('selena', 'renamed')
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
})

it('does not retire gateways when validation rejects the submit', async () => {
  render(<RenameProfileDialog currentName="selena" onClose={vi.fn()} open />)

  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  await waitFor(() => expect(screen.getByText('Name is required.')).toBeTruthy())
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(renameProfile).not.toHaveBeenCalled()
})
