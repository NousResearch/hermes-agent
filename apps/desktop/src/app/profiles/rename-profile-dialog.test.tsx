import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ProfileScope, setApiRequestConnection, setApiRequestLocalMode } from '@/api/client'
import type { HermesConnection } from '@/global'
import { renameProfile } from '@/hermes'
import { retireLocalProfileGateways } from '@/store/gateway'
import { $connection } from '@/store/session'
import { migrateTilesForProfile } from '@/store/session-states'
import { LEGACY_REMOTE_APPEARANCE_SCOPE, modePref, skinPref } from '@/themes/appearance-prefs'

import { RenameProfileDialog } from './rename-profile-dialog'

// Pins the rename half of the deleted-profile-resurrection class (#88638 fixed
// the delete half): a retained renderer socket for the OLD profile name must be
// retired BEFORE the rename PATCH tears down its backend, or the socket's
// reconnect loop respawns the old-name backend and recreates the directory the
// rename just moved.

beforeEach(() => {
  window.localStorage.clear()
  setApiRequestConnection(null)
  setApiRequestLocalMode(true)
  $connection.set({ connectionId: 'local', profile: 'default' } as HermesConnection)
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  setApiRequestConnection(null)
  setApiRequestLocalMode(true)
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
})

it('moves appearance on an explicitly renamed remote gateway only', async () => {
  window.localStorage.clear()
  skinPref.assign('selena', 'catppuccin', 'local')
  skinPref.assign('selena', 'ember', 'remote')
  modePref.assign('selena', 'dark', 'remote')
  skinPref.assign('other', 'nous', 'remote')
  modePref.assign('other', 'light', 'remote')
  render(
    <RenameProfileDialog
      currentName="selena"
      onClose={vi.fn()}
      open
      scope={{ connectionId: 'remote', profile: 'selena' }}
    />
  )
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))
  await waitFor(() => expect(skinPref.resolve('renamed', 'remote')).toBe('ember'))
  expect(modePref.resolve('renamed', 'remote')).toBe('dark')
  expect(skinPref.resolve('selena', 'local')).toBe('catppuccin')
  expect(skinPref.resolve('selena', 'remote')).toBe('nous')
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
  window.localStorage.clear()
})

it.each([
  { display: 'Sung_', scope: undefined },
  { display: 'Sung', scope: { connectionId: 'remote', profile: 'default' } }
])('keeps canonical default appearance when only its display label becomes $display', async ({ display, scope }) => {
  window.localStorage.clear()
  skinPref.assign('default', 'catppuccin', 'local')
  skinPref.assign('default', 'nous', 'remote')
  modePref.assign('default', 'dark', 'local')
  modePref.assign('default', 'light', 'remote')
  const skins = window.localStorage.getItem('hermes-desktop-profile-themes-v2')
  const modes = window.localStorage.getItem('hermes-desktop-profile-modes-v2')
  const onRenamed = vi.fn()
  render(
    <RenameProfileDialog currentName="default" isDefault onClose={vi.fn()} onRenamed={onRenamed} open scope={scope} />
  )
  fireEvent.change(screen.getByRole('textbox'), { target: { value: display } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))
  await waitFor(() => expect(onRenamed).toHaveBeenCalledWith(display))
  expect(window.localStorage.getItem('hermes-desktop-profile-themes-v2')).toBe(skins)
  expect(window.localStorage.getItem('hermes-desktop-profile-modes-v2')).toBe(modes)
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  window.localStorage.clear()
})

it('renames the active remote appearance when the caller omits scope', async () => {
  skinPref.assign('work', 'catppuccin', 'local')
  skinPref.assign('other', 'nous', 'local')
  skinPref.assign('work', 'ember', 'remote')
  modePref.assign('work', 'dark', 'remote')
  skinPref.assign('other', 'midnight', 'remote')
  modePref.assign('other', 'light', 'remote')
  $connection.set({ connectionId: 'remote', profile: 'default', mode: 'remote' } as HermesConnection)
  setApiRequestConnection('remote')
  setApiRequestLocalMode(false)
  const onRenamed = vi.fn()
  render(<RenameProfileDialog currentName="work" onClose={vi.fn()} onRenamed={onRenamed} open />)
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'studio' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))
  await waitFor(() => expect(onRenamed).toHaveBeenCalledWith('studio'))
  expect(skinPref.resolve('studio', 'remote')).toBe('ember')
  expect(modePref.resolve('studio', 'remote')).toBe('dark')
  expect(skinPref.resolve('work', 'remote')).toBe('midnight')
  expect(skinPref.resolve('work', 'local')).toBe('catppuccin')
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect(migrateTilesForProfile).not.toHaveBeenCalled()
})

it.each<[string, null | string, ProfileScope, string]>([
  ['registered ambient', 'remote', undefined, 'remote'],
  ['legacy ambient', null, undefined, LEGACY_REMOTE_APPEARANCE_SCOPE],
  ['registered profile alias', 'remote', 'alias', 'remote'],
  ['legacy profile alias', null, 'alias', LEGACY_REMOTE_APPEARANCE_SCOPE],
  ['registered object alias', 'remote', { profile: 'alias' }, 'remote'],
  ['legacy object alias', null, { connectionId: '  ', profile: 'alias' }, LEGACY_REMOTE_APPEARANCE_SCOPE]
])(
  'renames the %s request owner during a null-descriptor reconnect, not its local namesake',
  async (_label, connectionId, scope, owner) => {
    skinPref.assign('work', 'catppuccin', 'local')
    modePref.assign('work', 'system', 'local')
    skinPref.assign('studio', 'nous', 'local')
    modePref.assign('studio', 'light', 'local')
    skinPref.assign('work', 'ember', owner)
    modePref.assign('work', 'dark', owner)
    skinPref.assign('other', 'midnight', owner)
    modePref.assign('other', 'light', owner)
    setApiRequestConnection(connectionId)
    setApiRequestLocalMode(false)
    $connection.set(null)
    let release!: () => void

    const gate = new Promise<void>(resolve => {
      release = resolve
    })

    vi.mocked(renameProfile).mockImplementationOnce(async () => {
      await gate

      return { name: 'studio', ok: true, path: '/profiles/studio' }
    })
    const onRenamed = vi.fn()
    render(<RenameProfileDialog currentName="work" onClose={vi.fn()} onRenamed={onRenamed} open scope={scope} />)
    fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'studio' } })
    fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))
    await waitFor(() => expect(renameProfile).toHaveBeenCalled())
    expect(vi.mocked(renameProfile).mock.calls[0]).toEqual(
      scope == null ? ['work', 'studio'] : ['work', 'studio', scope]
    )
    setApiRequestConnection('local')
    setApiRequestLocalMode(true)
    $connection.set({ connectionId: 'local', profile: 'default', mode: 'local' } as HermesConnection)
    release()
    await waitFor(() => expect(onRenamed).toHaveBeenCalledWith('studio'))
    expect([skinPref.resolve('studio', owner), modePref.resolve('studio', owner)]).toEqual(['ember', 'dark'])
    expect([skinPref.resolve('work', owner), modePref.resolve('work', owner)]).toEqual(['midnight', 'light'])
    expect([skinPref.resolve('work', 'local'), modePref.resolve('work', 'local')]).toEqual(['catppuccin', 'system'])
    expect([skinPref.resolve('studio', 'local'), modePref.resolve('studio', 'local')]).toEqual(['nous', 'light'])
    expect(retireLocalProfileGateways).not.toHaveBeenCalled()
    expect(migrateTilesForProfile).not.toHaveBeenCalled()
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
