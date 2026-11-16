import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getHermesConfig } from '@/api/config'
import { $desktopBoot } from '@/store/boot'
import { closeSecondaryGateways } from '@/store/gateway'
import { $activeGatewayProfile, $profiles, invalidateProfileListFetches, refreshProfiles } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { $settingsRequestProfile } from '@/store/settings-scope'

import { PrimaryProfilePin } from '../../../../electron/primary-profile-pin'
import { connectPrimaryRemote } from '../../../../electron/primary-remote-startup'
import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

// Cross the Electron restore boundary before booting the real renderer stores
// and config helper. Only the IPC/remote transport and WebSocket are replaced.
vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

const originalWebSocket = globalThis.WebSocket

function closeGateways() {
  takeGatewaySurvivor()?.gateway.close()
  closeSecondaryGateways()
}

beforeEach(() => {
  closeGateways()
  invalidateProfileListFetches()
  $activeGatewayProfile.set('default')
  $profiles.set([])
  $connection.set(null)
  $gatewayState.set('idle')
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.pingMode = 'pong'
  FakeWebSocket.instances = []
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: '',
    phase: 'init',
    progress: 0,
    running: true,
    timestamp: Date.now(),
    visible: true
  })
})

afterEach(() => {
  cleanup()
  closeGateways()
  invalidateProfileListFetches()
  $connection.set(null)
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

it.each(['default', 'deleted-workspace'])(
  'loads settings on the same gateway after restoring %s from workspace memory',
  async remembered => {
    let rememberedProfile = remembered
    const pin = new PrimaryProfilePin()
    pin.pin(rememberedProfile)
    const requests: Array<{ path: string; profile?: string; connectionId?: string }> = []

    const api = vi.fn(async request => {
      requests.push(request)

      if (request.path === '/api/profiles') {
        return {
          profiles: [
            {
              name: 'default',
              is_default: true,
              has_env: false,
              model: null,
              provider: null,
              path: '/offline/default',
              skill_count: 0
            }
          ]
        }
      }

      if (request.path === '/api/config') {
        if (request.profile && request.profile !== 'default') {
          throw new Error(`404 unknown profile: ${request.profile}`)
        }

        return { model: 'offline-model' }
      }

      return {}
    })

    const connection = await connectPrimaryRemote(
      {
        baseUrl: 'https://offline-gateway.invalid',
        wsUrl: 'wss://offline-gateway.invalid/api/ws',
        connectionId: 'offline-remote',
        source: 'settings',
        token: 'offline-placeholder'
      },
      {
        profile: pin.resolve(() => rememberedProfile),
        restoreRememberedProfile: () => true,
        assertCurrentAttempt: () => undefined,
        advanceBootProgress: async () => undefined,
        waitForRemote: async () => undefined,
        readProfiles: remote => api({ path: '/api/profiles', connectionId: remote.connectionId }),
        onProfileRestored: profile => {
          pin.pin(profile)
          rememberedProfile = profile
        },
        updateBootProgress: () => undefined,
        logs: () => [],
        windowState: () => ({})
      }
    )

    const unsubscribe = () => () => undefined

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      api,
      getConnection: async () => connection,
      getGatewayWsUrl: async () => connection.wsUrl,
      getBootProgress: async () => ({ running: true, phase: 'init', progress: 0 }),
      onBootProgress: unsubscribe,
      onBackendExit: unsubscribe,
      onConnectionApplied: unsubscribe,
      onPowerResume: unsubscribe,
      onWindowStateChanged: unsubscribe,
      profile: {
        get: async () => ({ profile: rememberedProfile }),
        getDefault: async () => null
      }
    }
    let configError: unknown

    function Harness() {
      useGatewayBoot({
        beforeConnectionSwitch: () => undefined,
        handleGatewayEvent: () => undefined,
        handleServerRequest: () => false,
        onConnectionReady: () => undefined,
        onGatewayReady: () => undefined,
        refreshSessions: async () => undefined,
        refreshHermesConfig: async () => {
          // Establish a successful authoritative roster, not an empty cache.
          await refreshProfiles()
          await getHermesConfig($settingsRequestProfile.get()).catch(error => {
            configError = error
          })
        }
      })

      return null
    }

    render(<Harness />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().running).toBe(false)
    expect($profiles.get().map(profile => profile.name)).toEqual(['default'])
    expect(requests.find(request => request.path === '/api/config')).toMatchObject({
      connectionId: 'offline-remote',
      profile: 'default'
    })
    expect($activeGatewayProfile.get()).toBe('default')
    expect(pin.booted).toBe('default')
    expect(rememberedProfile).toBe('default')
    expect(configError).toBeUndefined()
  }
)
