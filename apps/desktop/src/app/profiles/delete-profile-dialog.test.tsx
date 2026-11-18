import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type * as HermesApi from '@/hermes'
import { deleteProfile } from '@/hermes'
import type * as GatewayStore from '@/store/gateway'
import { retireLocalProfileGateways } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { setConnection } from '@/store/session'
import { $sessionTiles, openSessionTile } from '@/store/session-states'

import { DeleteProfileDialog } from './delete-profile-dialog'

vi.mock('@/hermes', async importOriginal => ({ ...(await importOriginal<typeof HermesApi>()), deleteProfile: vi.fn() }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  retireLocalProfileGateways: vi.fn()
}))
vi.mock('@/components/ui/confirm-dialog', () => ({
  ConfirmDialog: ({ onConfirm }: { onConfirm: () => Promise<void> }) => (
    <button onClick={() => void onConfirm()}>Delete</button>
  )
}))
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

const local = (profile: string) =>
  ({ baseUrl: 'http://localhost:8000', connectionId: 'local', mode: 'local', profile }) as HermesConnection

async function confirmDelete(element: React.ReactElement) {
  vi.mocked(deleteProfile).mockResolvedValueOnce({ ok: true, path: '/agent' })
  render(element)
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Delete' })))
}

it('drops a Bot tile that targets the deleted profile under another name when the delete has no owner', async () => {
  setConnection(local('default'))
  $sessionTiles.set([])
  openSessionTile('alias-bot', 'right', undefined, undefined, {
    workspaceMode: 'bots',
    ownerRoute: { connectionId: 'local', profile: 'alias', targetProfile: 'doomed' }
  })
  await confirmDelete(<DeleteProfileDialog onClose={() => {}} open profile={{ name: 'doomed', path: '/agent' }} />)
  expect(deleteProfile).toHaveBeenCalledWith('doomed')
  // Left behind, it restores on relaunch and dials the deleted profile's backend back to life.
  expect($sessionTiles.get()).toEqual([])
})

it('treats any scope as another gateway: nothing local retired and the window keeps its agent', async () => {
  setConnection(local('doomed'))
  $activeGatewayProfile.set('doomed')
  await confirmDelete(
    <DeleteProfileDialog
      onClose={() => {}}
      open
      profile={{ name: 'doomed', path: '/agent' }}
      scope={{ connectionId: 'local', profile: 'doomed' }}
    />
  )
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  expect($activeGatewayProfile.get()).toBe('doomed')
})

it('retires a local owner and moves the window off the agent it deleted', async () => {
  setConnection(local('doomed'))
  $activeGatewayProfile.set('doomed')
  await confirmDelete(
    <DeleteProfileDialog
      onClose={() => {}}
      open
      owner={{ connectionId: 'local', profile: 'doomed' }}
      profile={{ name: 'doomed', path: '/agent' }}
    />
  )
  expect(retireLocalProfileGateways).toHaveBeenCalledWith('doomed')
  expect(deleteProfile).toHaveBeenCalledWith('doomed', { connectionId: 'local', profile: 'doomed' })
  expect($activeGatewayProfile.get()).toBe('default')
})

it('deletes only the captured gateway owner when a same-named agent is foreground at completion', async () => {
  const profile = 'delete-owner-regression'
  const local = { baseUrl: 'http://localhost:8000', connectionId: 'local', mode: 'local', profile } as HermesConnection

  const remote = {
    baseUrl: 'https://remote.example',
    connectionId: 'remote',
    mode: 'remote',
    profile
  } as HermesConnection

  setConnection(local)
  $activeGatewayProfile.set(profile)
  $sessionTiles.set([])
  openSessionTile('keep-local', 'right', undefined, undefined, {
    workspaceMode: 'sessions',
    ownerRoute: { connectionId: 'local', profile }
  })
  setConnection(remote)
  openSessionTile('delete-remote', 'right', undefined, undefined, {
    workspaceMode: 'sessions',
    ownerRoute: { connectionId: 'remote', profile }
  })
  const { promise, resolve: finish } = Promise.withResolvers<{ ok: boolean; path: string }>()
  vi.mocked(deleteProfile).mockReturnValueOnce(promise)
  render(
    <DeleteProfileDialog
      onClose={() => {}}
      open
      profile={{ name: profile, path: '/remote/agent' }}
      scope={{ connectionId: 'remote', profile }}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
  setConnection(local)
  await act(async () => finish({ ok: true, path: '/remote/agent' }))
  expect(deleteProfile).toHaveBeenCalledWith(profile, { connectionId: 'remote', profile })
  expect($activeGatewayProfile.get()).toBe(profile)
  expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['keep-local'])
  setConnection(remote)
  expect($sessionTiles.get()).toEqual([])
  setConnection(local)
  expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['keep-local'])
})
