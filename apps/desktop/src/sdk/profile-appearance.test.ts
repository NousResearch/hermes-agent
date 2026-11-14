import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { deleteProfile } from '@/hermes'
import { activeGatewayConnectionId } from '@/store/gateway'
import { $connection } from '@/store/session'
import { modePref, skinPref } from '@/themes/appearance-prefs'

import { host } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<object>()),
  deleteProfile: vi.fn(),
  getProfiles: vi.fn(async () => ({ profiles: [] }))
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<object>()),
  activeGatewayConnectionId: vi.fn(),
  retireLocalProfileGateways: vi.fn()
}))

beforeEach(() => {
  window.localStorage.clear()
})

afterEach(() => {
  $connection.set(null)
  vi.clearAllMocks()
})

it.each([
  ['local', 'remote'],
  ['remote', 'local']
])('keeps an in-flight SDK %s delete bound to its initiating gateway after switching to %s', async (owner, peer) => {
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
  vi.mocked(activeGatewayConnectionId).mockReturnValue(owner)
  const pending = host.deleteProfile('work')
  expect(deleteProfile).toHaveBeenCalled()
  $connection.set({
    connectionId: peer,
    mode: peer === 'local' ? 'local' : 'remote',
    profile: 'default'
  } as HermesConnection)
  vi.mocked(activeGatewayConnectionId).mockReturnValue(peer)
  release()
  await pending
  expect([skinPref.resolve('work', owner), modePref.resolve('work', owner)]).toEqual(['midnight', 'light'])
  expect([skinPref.resolve('work', peer), modePref.resolve('work', peer)]).toEqual(['ember', 'dark'])
})
