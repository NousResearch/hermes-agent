import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { $activeGatewayProfile } from '@/store/profile'

// Voice mode and the Appearance display toggles write `config.set` for the
// route the window is on. A profile served only by a remote gateway (`gamma` on
// Box) has no backend on this machine: resolving its bare name through the
// profile pool asked main for a LOCAL backend of that name (refused, or worse,
// This device's same-named profile), so the write never reached Box.

const sockets = vi.hoisted(() => [] as Array<{ requests: Array<[string, unknown]>; wsUrl: string }>)

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  HermesGateway: class {
    connectionState = 'closed'
    record = { requests: [] as Array<[string, unknown]>, wsUrl: '' }
    constructor() {
      sockets.push(this.record)
    }
    connect = async (wsUrl: string): Promise<void> => {
      this.record.wsUrl = wsUrl
      this.connectionState = 'open'
    }
    request = async (method: string, params: unknown) => {
      this.record.requests.push([method, params])

      return {}
    }
    close = (): void => {
      this.connectionState = 'closed'
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))

vi.mock('@/lib/voice-live', () => ({ fetchVoiceLiveStatus: async () => null }))

const { closeSecondaryGateways, configureGatewayRegistry, ensureGatewayForAgent, setPrimaryGateway } =
  await import('@/store/gateway')

const { setVoiceChatMode } = await import('./voice-live')
const { mirrorDisplayToggle } = await import('./display-toggles')

const $toggle = atom(true)
mirrorDisplayToggle('display.in_app_tips', 'hermes.desktop.route-test-toggle.v1', $toggle)

const boxGamma = {
  authMode: 'token',
  baseUrl: 'http://box.invalid',
  connectionId: 'box',
  mode: 'remote',
  profile: 'gamma',
  token: 'fake-test-token',
  wsUrl: 'ws://box.invalid/api/ws?token=fake-test-token'
}

afterEach(() => {
  closeSecondaryGateways()
  sockets.length = 0
  $activeGatewayProfile.set('default')
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('config writes on a remote-only profile', () => {
  it('voice mode and display toggles land on the remote route, never a local backend of that name', async () => {
    const getConnection = vi.fn(async (profile?: string) => {
      if (profile) {
        throw new Error(`Profile "${profile}" no longer exists.`)
      }

      return { mode: 'local', port: 4242, token: 'primary-token' }
    })

    const primaryRequest = vi.fn(async () => ({}))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection,
      getConnectionFor: vi.fn(async () => boxGamma)
    }

    configureGatewayRegistry({ onActiveRouteChanged: profile => $activeGatewayProfile.set(profile), onEvent: vi.fn() })
    setPrimaryGateway({ connectionState: 'open', request: primaryRequest } as never, 'default')
    await expect(ensureGatewayForAgent('box', 'gamma')).resolves.toBe(true)

    await setVoiceChatMode('gpt-live')
    $toggle.set(false)

    // Box's dedicated gamma socket: the write is scoped by the socket itself.
    const box = sockets.find(socket => socket.wsUrl === boxGamma.wsUrl)

    await vi.waitFor(() =>
      expect(box?.requests).toEqual(
        expect.arrayContaining([
          ['config.set', { key: 'voice.voice_chat_mode', profile: 'gamma', value: 'gpt-live' }],
          ['config.set', { key: 'display.in_app_tips', profile: 'gamma', value: 'false' }]
        ])
      )
    )
    expect(getConnection).not.toHaveBeenCalledWith('gamma')
    expect(primaryRequest).not.toHaveBeenCalledWith('config.set', expect.anything())
  })

  it('a saved toggle re-sent on a gateway switch lands on the new route only (local/default -> box/gamma)', async () => {
    const primaryRequest = vi.fn(async () => ({}))

    const getConnectionFor = vi.fn(async ({ profile }: { connectionId: string; profile: string }) => ({
      ...boxGamma,
      profile,
      wsUrl: profile === 'gamma' ? boxGamma.wsUrl : `ws://box.invalid/api/ws?profile=${profile}`
    }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async () => ({ mode: 'local', port: 4242, token: 'primary-token' })),
      getConnectionFor
    }

    configureGatewayRegistry({ onActiveRouteChanged: profile => $activeGatewayProfile.set(profile), onEvent: vi.fn() })
    setPrimaryGateway({ connectionState: 'open', request: primaryRequest } as never, 'default')
    // The user touched the toggle earlier: every socket change re-sends it.
    localStorage.setItem('hermes.desktop.route-test-toggle.v1', 'true')

    try {
      await expect(ensureGatewayForAgent('box', 'gamma')).resolves.toBe(true)

      const box = sockets.find(socket => socket.wsUrl === boxGamma.wsUrl)

      await vi.waitFor(() =>
        expect(box?.requests).toContainEqual([
          'config.set',
          { key: 'display.in_app_tips', profile: 'gamma', value: String($toggle.get()) }
        ])
      )
      expect(getConnectionFor).not.toHaveBeenCalledWith(expect.objectContaining({ profile: 'default' }))
      expect(sockets.filter(socket => socket !== box).flatMap(socket => socket.requests)).toEqual([])
      expect(primaryRequest).not.toHaveBeenCalledWith('config.set', expect.anything())
    } finally {
      localStorage.removeItem('hermes.desktop.route-test-toggle.v1')
    }
  })
})
