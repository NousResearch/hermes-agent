import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ProfileScope, setApiRequestConnection, setApiRequestLocalMode } from '@/api/client'
import type { HermesConnection } from '@/global'
import { deleteProfile } from '@/hermes'
import { $connection } from '@/store/session'
import { LEGACY_REMOTE_APPEARANCE_SCOPE, modePref, skinPref } from '@/themes/appearance-prefs'

import { DeleteProfileDialog } from './delete-profile-dialog'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<object>()),
  deleteProfile: vi.fn(async () => ({ ok: true }))
}))

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

it('deletes the active remote appearance when the caller omits scope', async () => {
  skinPref.assign('work', 'catppuccin', 'local')
  skinPref.assign('other', 'nous', 'local')
  skinPref.assign('work', 'ember', 'remote')
  modePref.assign('work', 'dark', 'remote')
  skinPref.assign('other', 'midnight', 'remote')
  modePref.assign('other', 'light', 'remote')
  $connection.set({ connectionId: 'remote', profile: 'default', mode: 'remote' } as HermesConnection)
  setApiRequestConnection('remote')
  setApiRequestLocalMode(false)
  const onDeleted = vi.fn()
  render(
    <DeleteProfileDialog
      onClose={vi.fn()}
      onDeleted={onDeleted}
      open
      profile={{ name: 'work', path: '/profiles/work' }}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: /^delete$/i }))
  await waitFor(() => expect(onDeleted).toHaveBeenCalled())
  expect(deleteProfile).toHaveBeenCalledWith('work')
  expect(skinPref.resolve('work', 'local')).toBe('catppuccin')
  expect(skinPref.resolve('work', 'remote')).toBe('midnight')
  expect(modePref.resolve('work', 'remote')).toBe('light')
})

it.each([
  ['local', 'remote'],
  ['remote', 'local']
])('keeps an in-flight %s delete bound to its initiating gateway after switching to %s', async (owner, peer) => {
  for (const connectionId of [owner, peer]) {
    skinPref.assign('work', 'ember', connectionId)
    modePref.assign('work', 'dark', connectionId)
    skinPref.assign('other', 'midnight', connectionId)
    modePref.assign('other', 'light', connectionId)
  }

  let release!: () => void

  const gate = new Promise<void>(resolve => {
    release = resolve
  })

  vi.mocked(deleteProfile).mockImplementationOnce(async () => {
    await gate

    return { ok: true, path: '/profiles/work' }
  })
  $connection.set({
    connectionId: owner,
    mode: owner === 'local' ? 'local' : 'remote',
    profile: 'default'
  } as HermesConnection)
  setApiRequestConnection(owner)
  setApiRequestLocalMode(owner === 'local')
  const onDeleted = vi.fn()
  render(
    <DeleteProfileDialog
      onClose={vi.fn()}
      onDeleted={onDeleted}
      open
      profile={{ name: 'work', path: '/profiles/work' }}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: /^delete$/i }))
  await waitFor(() => expect(deleteProfile).toHaveBeenCalledWith('work'))
  $connection.set({
    connectionId: peer,
    mode: peer === 'local' ? 'local' : 'remote',
    profile: 'default'
  } as HermesConnection)
  setApiRequestConnection(peer)
  setApiRequestLocalMode(peer === 'local')
  release()
  await waitFor(() => expect(onDeleted).toHaveBeenCalled())
  expect([skinPref.resolve('work', owner), modePref.resolve('work', owner)]).toEqual(['midnight', 'light'])
  expect([skinPref.resolve('work', peer), modePref.resolve('work', peer)]).toEqual(['ember', 'dark'])
})

it.each<[string, null | string, ProfileScope, string]>([
  ['registered ambient', 'remote', undefined, 'remote'],
  ['legacy ambient', null, undefined, LEGACY_REMOTE_APPEARANCE_SCOPE],
  ['registered profile alias', 'remote', 'alias', 'remote'],
  ['legacy profile alias', null, 'alias', LEGACY_REMOTE_APPEARANCE_SCOPE],
  ['registered object alias', 'remote', { profile: 'alias' }, 'remote'],
  ['legacy object alias', null, { connectionId: '  ', profile: 'alias' }, LEGACY_REMOTE_APPEARANCE_SCOPE]
])(
  'deletes the %s request owner during a null-descriptor reconnect, not its local namesake',
  async (_label, connectionId, scope, owner) => {
    skinPref.assign('work', 'catppuccin', 'local')
    modePref.assign('work', 'system', 'local')
    skinPref.assign('other', 'nous', 'local')
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

    vi.mocked(deleteProfile).mockImplementationOnce(async () => {
      await gate

      return { ok: true, path: '/profiles/work' }
    })
    const onDeleted = vi.fn()
    render(
      <DeleteProfileDialog
        onClose={vi.fn()}
        onDeleted={onDeleted}
        open
        profile={{ name: 'work', path: '/profiles/work' }}
        scope={scope}
      />
    )
    fireEvent.click(screen.getByRole('button', { name: /^delete$/i }))
    await waitFor(() => expect(deleteProfile).toHaveBeenCalled())
    expect(vi.mocked(deleteProfile).mock.calls[0]).toEqual(scope == null ? ['work'] : ['work', scope])
    setApiRequestConnection('local')
    setApiRequestLocalMode(true)
    $connection.set({ connectionId: 'local', profile: 'default', mode: 'local' } as HermesConnection)
    release()
    await waitFor(() => expect(onDeleted).toHaveBeenCalled())
    expect([skinPref.resolve('work', owner), modePref.resolve('work', owner)]).toEqual(['midnight', 'light'])
    expect([skinPref.resolve('work', 'local'), modePref.resolve('work', 'local')]).toEqual(['catppuccin', 'system'])
  }
)

it('deletes a background remote appearance without erasing the foreground local namesake', async () => {
  skinPref.assign('work', 'catppuccin', 'local')
  skinPref.assign('other', 'nous', 'local')
  skinPref.assign('work', 'ember', 'remote')
  modePref.assign('work', 'dark', 'remote')
  skinPref.assign('other', 'midnight', 'remote')
  modePref.assign('other', 'light', 'remote')
  const onDeleted = vi.fn()
  render(
    <DeleteProfileDialog
      onClose={vi.fn()}
      onDeleted={onDeleted}
      open
      profile={{ name: 'work', path: '/profiles/work' }}
      scope={{ connectionId: 'remote', profile: 'work' }}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: /^delete$/i }))
  await waitFor(() => expect(onDeleted).toHaveBeenCalled())
  expect(deleteProfile).toHaveBeenCalledWith('work', { connectionId: 'remote', profile: 'work' })
  expect(skinPref.resolve('work', 'local')).toBe('catppuccin')
  expect(skinPref.resolve('work', 'remote')).toBe('midnight')
  expect(modePref.resolve('work', 'remote')).toBe('light')
})
