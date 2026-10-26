import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { $connectionsRegistry, _resetConnectionsForTests } from '@/store/connections'
import { closeSecondaryGateways } from '@/store/gateway'
import { endGatewaySwitch } from '@/store/gateway-switch'
import { clearNotifications, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import {
  $awaitingResponse,
  $busy,
  $connection,
  $currentCwd,
  $gatewayState,
  setActiveSessionId,
  setSelectedStoredSessionId
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { warnIfTerminalBackendUnavailable } from '@/store/terminal-backend-warning'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

export let connectionApplied: null | (() => void) = null
export let powerResume: null | (() => void) = null
export let backendExit: null | ((payload?: unknown) => void) = null
export const primaryConn = {
  authMode: 'token' as 'oauth' | 'token',
  baseUrl: 'https://vps.example.com',
  connectionId: 'primary-vps',
  profile: 'default',
  token: 't',
  wsUrl: 'wss://vps.example.com/api/ws?token=t'
}

export const remotePrimaryConn = { ...primaryConn, mode: 'remote' as const }

export const coderConn = {
  authMode: 'token' as const,
  baseUrl: 'https://coder.example.com',
  connectionId: 'coder-remote',
  profile: 'coder',
  token: 'c',
  wsUrl: 'wss://coder.example.com/api/ws?token=c'
}

export function fakeDesktop() {
  let bootProgressHandler: ((payload: Record<string, unknown>) => void) | null = null

  return {
    getConnection: vi.fn(async (profile?: null | string) => {
      const key = (profile ?? '').trim()

      return !key || key === 'default' ? primaryConn : coderConn
    }),
    getGatewayWsUrl: vi.fn(async (conn?: { wsUrl?: string }) => conn?.wsUrl ?? primaryConn.wsUrl),
    getConnectionConfig: vi.fn(async () => ({ mode: 'local' })),
    getBootProgress: vi.fn(async () => ({
      error: null as null | string,
      fakeMode: false,
      message: '',
      phase: 'init',
      progress: 0,
      retryable: false as boolean,
      running: true as boolean,
      timestamp: Date.now()
    })),
    onBootProgress: vi.fn(callback => {
      bootProgressHandler = callback

      return () => {
        bootProgressHandler = null
      }
    }),
    // Test helper: fire a post-boot progress event through the real subscription.
    emitBootProgress(payload: Record<string, unknown>) {
      bootProgressHandler?.(payload)
    },
    onBackendExit: vi.fn(callback => {
      backendExit = callback

      return () => {
        backendExit = null
      }
    }),
    onConnectionApplied: vi.fn(callback => {
      connectionApplied = callback

      return () => {
        connectionApplied = null
      }
    }),
    onPowerResume: vi.fn(callback => {
      powerResume = callback

      return () => {
        powerResume = null
      }
    }),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    onWindowStateChanged: vi.fn(() => () => undefined),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

export function Harness({
  beforeConnectionSwitch = () => undefined,
  refreshHermesConfig = async () => undefined,
  refreshSessions
}: {
  beforeConnectionSwitch?: () => void
  refreshHermesConfig?: (force?: boolean, shouldPublish?: () => boolean) => Promise<void>
  refreshSessions?: (shouldPublish?: () => boolean) => Promise<void>
} = {}) {
  useGatewayBoot({
    beforeConnectionSwitch,
    handleGatewayEvent: () => undefined,
    handleServerRequest: () => false,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig,
    refreshSessions: refreshSessions ?? (async () => undefined)
  })

  return null
}

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
  // Drop any parked gateway left by a prior file/case (globalThis slot).
  const leftover = takeGatewaySurvivor()

  if (leftover) {
    try {
      leftover.gateway.close()
    } catch {
      // ignore
    }
  }

  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $profiles.set([])
  $sessionTiles.set([])
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  connectionApplied = null
  powerResume = null
  backendExit = null
  clearNotifications()
  vi.mocked(notifyError).mockReset()
  vi.mocked(warnIfTerminalBackendUnavailable).mockClear()
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = fakeDesktop()
  $gatewayState.set('idle')
  $busy.set(false)
  $awaitingResponse.set(false)
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
  // Vitest keeps import.meta.hot truthy, so the boot effect's cleanup parks an
  // open gateway instead of tearing it down (the real HMR path). Drain + close
  // that survivor so the next test boots a fresh socket instead of adoptBoot().
  const survivor = takeGatewaySurvivor()

  if (survivor) {
    try {
      survivor.gateway.close()
    } catch {
      // ignore
    }
  }

  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $profiles.set([])
  $sessionTiles.set([])
  _resetConnectionsForTests()
  $connectionsRegistry.set(null)
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  endGatewaySwitch()
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  window.localStorage.removeItem('hermes.desktop.workspace-cwd')
  $currentCwd.set('')
  $busy.set(false)
  $awaitingResponse.set(false)
})

// Let pending microtasks (awaits) AND the queued 0ms socket open/error fire.
export async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}
