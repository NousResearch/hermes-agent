import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type * as HermesApi from '@/hermes'
import { deleteProfile } from '@/hermes'
import { $activeGatewayProfile } from '@/store/profile'
import { setConnection } from '@/store/session'
import { $sessionTiles, openSessionTile } from '@/store/session-states'

import { DeleteProfileDialog } from './delete-profile-dialog'

vi.mock('@/hermes', async importOriginal => ({ ...(await importOriginal<typeof HermesApi>()), deleteProfile: vi.fn() }))
vi.mock('@/components/ui/confirm-dialog', () => ({
  ConfirmDialog: ({ onConfirm }: { onConfirm: () => Promise<void> }) => (
    <button onClick={() => void onConfirm()}>Delete</button>
  )
}))
afterEach(cleanup)

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
