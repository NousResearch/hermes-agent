import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'


// Registry-agent activation (ensureGatewayAgent — the SDK ensureAgent door).
// Two regressions pinned here:
//  1. Activating an ALREADY-OPEN registry agent must still resync
//     $connection (via getConnectionFor) and move $activeGatewayProfile —
//     previously only a freshly-dialed socket synced $connection (inside
//     openSecondary), so re-activating an open agent left REST/fs/media and
//     image-attach routing on the previous backend (same class as #46651).
//  2. Agent activations share the gatewaySwitch mutex with profile switches —
//     without it, two rapid activations could complete out of order and the
//     EARLIER setActive() landed last.

const ensureGatewayForAgent = vi.fn(async (_connectionId: null | string, _profile: string) => true)
const ensureGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const openGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const $gateway = atom<unknown>({ id: 'live-socket' })
const resetStarmapGraph = vi.fn()
const activeGatewayProfileKey = vi.fn(() => 'default')
const activeGatewayConnectionId = vi.fn(() => 'local')

vi.mock('@/store/gateway', () => ({
  $gateway,
  activeGatewayProfileKey,
  activeGatewayConnectionId,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph }))

const { $activeGatewayProfile, $gatewaySwapTarget, ensureGatewayAgent } = await import('./profile')

const {
  $connection,
  setComposerSelectionOwner
} = await import('./session')

const localConn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({ baseUrl: '', mode: 'local', profile: 'default', ...over }) as HermesConnection

const getConnection = vi.fn<(profile?: string | null) => Promise<HermesConnection>>()

const getConnectionFor =
  vi.fn<(payload: { connectionId?: null | string; profile?: null | string }) => Promise<HermesConnection>>()

beforeEach(() => {
  const localStorage = window.localStorage

  localStorage.clear()
  getConnection.mockReset()
  getConnectionFor.mockReset()
  ensureGatewayForAgent.mockClear()
  activeGatewayProfileKey.mockReturnValue('default')
  activeGatewayConnectionId.mockReturnValue('local')
  ensureGatewayForProfile.mockClear()
  $gateway.set({ id: 'live-socket' })
  $activeGatewayProfile.set('default')
  $connection.set(localConn())
  vi.stubGlobal('window', { hermesDesktop: { getConnection, getConnectionFor }, localStorage })
  setComposerSelectionOwner('homelab', 'default')
})

afterEach(() => {
  vi.unstubAllGlobals()
  $connection.set(null)
})

describe('registry session activation on an already-ready shared host', () => {
  it('does not wake the same profile; switching A→B→A still activates each different owner', async () => {
    const descriptor = localConn({ connectionId: 'local', profile: 'research', sharedPrimary: true })
    const overlays: (null | string)[] = []
    const beforeActivate = vi.fn(() => true)
    $gateway.set({ id: 'live-socket', connectionState: 'open' })
    $activeGatewayProfile.set('research')
    $connection.set(descriptor)
    getConnectionFor.mockImplementation(async ({ profile }) =>
      localConn({ connectionId: 'local', profile: profile ?? 'default', sharedPrimary: true })
    )
    const unbind = $gatewaySwapTarget.listen(value => overlays.push(value))

    try {
      for (let index = 0; index < 3; index++) {
        await ensureGatewayAgent('local', 'research', { beforeActivate })
      }

      expect(beforeActivate).toHaveBeenCalledTimes(3)
      expect(overlays).toEqual([])
      expect(ensureGatewayForAgent).not.toHaveBeenCalled()
      expect(getConnectionFor).not.toHaveBeenCalled()
      expect($connection.get()).toBe(descriptor)
      await ensureGatewayAgent('local', 'writer')
      await ensureGatewayAgent('local', 'research')
      expect(overlays).toEqual(['writer', null, 'research', null])
      expect(ensureGatewayForAgent.mock.calls).toEqual([['local', 'writer'], ['local', 'research']])
      expect($activeGatewayProfile.get()).toBe('research')
      expect($connection.get()?.profile).toBe('research')
    } finally {
      unbind()
    }
  })

  it('does not reuse a mismatched non-shared route or bypass the caller commit veto', async () => {
    $gateway.set({ id: 'live-socket', connectionState: 'open' })
    $activeGatewayProfile.set('research')
    $connection.set(localConn({ connectionId: 'local', profile: 'research' }))
    getConnectionFor.mockResolvedValue(localConn({ connectionId: 'local', profile: 'research' }))
    await ensureGatewayAgent('local', 'research')
    expect(ensureGatewayForAgent).toHaveBeenCalledOnce()
    ensureGatewayForAgent.mockClear()
    const veto = vi.fn(() => false)
    await ensureGatewayAgent('local', 'writer', { beforeActivate: veto })
    expect(veto).toHaveBeenCalledOnce()
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
    expect($activeGatewayProfile.get()).toBe('research')
  })
})
