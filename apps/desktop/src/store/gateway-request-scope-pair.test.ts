import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope, setApiRequestConnection, setApiRequestProfile } from '@/api/client'

// A profile belongs to ONE gateway: the ambient REST scope is a
// (connection, profile) pair, and Electron main dials whatever backend a
// request names. Every scope listener (the version poller among them) must only
// ever observe pairs that exist, never the new gateway with the old profile.

vi.mock(import('@/hermes'), async importOriginal => ({
  ...(await importOriginal()),
  HermesGateway: class {
    connectionState = 'closed'
    connect = async (): Promise<void> => {
      this.connectionState = 'open'
    }
    close = (): void => {
      this.connectionState = 'closed'
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  } as never
}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} = await import('./gateway')

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      authMode: 'token',
      connectionId,
      profile,
      token: 't',
      wsUrl: `wss://${connectionId}.invalid/api/ws?profile=${profile}`
    })),
    getGatewayWsUrlFor: vi.fn(
      async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        `wss://${connectionId}.invalid/api/ws?profile=${profile}`
    )
  }
}

let seen: string[] = []

let stopWatching = (): void => {}

beforeEach(() => {
  installDesktop()
  // The renderer's profile mirror (store/profile) follows the active route
  // into the request scope; reproduce that hop without the store's UI deps.
  configureGatewayRegistry({ onActiveRouteChanged: profile => setApiRequestProfile(profile), onEvent: vi.fn() })
  // Window primary: this device, profile alpha.
  setPrimaryGateway({ connectionState: 'open' } as never, 'alpha')
  setPrimaryGatewayConnectionId('local', 'local')
  setApiRequestProfile('alpha')
  seen = []
  stopWatching = $apiRequestScope.listen(({ connectionId, profile }) => {
    seen.push(`${connectionId}|${profile}`)
  })
})

afterEach(() => {
  stopWatching()
  closeSecondaryGateways()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ambient request scope across a gateway switch', () => {
  it('moves connection and profile together from (local, alpha) to (box, default) and back', async () => {
    await expect(ensureGatewayForAgent('box', 'default')).resolves.toBe(true)

    expect(seen).not.toContain('box|alpha')
    expect($apiRequestScope.get()).toEqual({ connectionId: 'box', profile: 'default' })

    await ensureGatewayForProfile('alpha')

    expect(seen).toEqual(['box|default', 'local|alpha'])
  })

  it('re-homing the foregrounded primary to another machine carries the primary profile with it', () => {
    setPrimaryGateway({ connectionState: 'open' } as never, 'default')
    setPrimaryGatewayConnectionId('box', 'remote')

    expect(seen).not.toContain('box|alpha')
    expect($apiRequestScope.get()).toEqual({ connectionId: 'box', profile: 'default' })
  })

  it('leaves the profile half alone while the connection stays put', () => {
    // A shared-primary socket serves profiles other than its route's own; the
    // profile mirror owns that half, so a same-connection re-publish keeps it.
    setApiRequestProfile('beta')
    setPrimaryGatewayConnectionId('local', 'local')

    expect($apiRequestScope.get()).toEqual({ connectionId: 'local', profile: 'beta' })
  })
})
