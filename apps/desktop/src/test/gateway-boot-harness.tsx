import type { GatewayEvent } from '@hermes/shared'
/**
 * Shared harness for the useGatewayBoot suites (#C2a).
 *
 * Drives the REAL hook + REAL HermesGateway through a fake WebSocket and a fake
 * Electron bridge — an end-to-end-ish repro from the renderer's point of view
 * (a "remote VPS" is just a WebSocket that opens once and later refuses to
 * reopen, so that is exactly what is faked).
 *
 * Extracted from use-gateway-boot.test.tsx so the wake contract can live in its
 * own suite (use-gateway-boot-wake.test.tsx) without duplicating any of this,
 * and without growing a file that is at its line cap. Each suite keeps its own
 * `vi.mock` declarations, which are per test file.
 */
import { act, cleanup } from '@testing-library/react'
import { vi } from 'vitest'

import { takeGatewaySurvivor } from '@/app/gateway/hooks/gateway-hmr-survivor'
import { useGatewayBoot } from '@/app/gateway/hooks/use-gateway-boot'
import type { DesktopConnectionsRegistry } from '@/global'
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

import { FakeWebSocket } from './fake-gateway-socket'

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

// Live bindings: fakeDesktop() below publishes each subscription's handler here,
// so a test can drive the exact callback the main process would.
export let connectionApplied: null | (() => void) = null
export let powerResume: null | (() => void) = null
export let backendExit: null | ((payload?: unknown) => void) = null

export function fakeDesktop() {
  let bootProgressHandler: ((payload: Record<string, unknown>) => void) | null = null

  return {
    // C1 telemetry sink: `reconnect-episode …` lines land here. Swapped by the
    // tests that read the phase table.
    logLine: vi.fn(),
    getConnection: vi.fn(async (profile?: null | string) => {
      const key = (profile ?? '').trim()

      return !key || key === 'default' ? primaryConn : coderConn
    }),
    getGatewayWsUrl: vi.fn(async (conn?: { wsUrl?: string }) => conn?.wsUrl ?? primaryConn.wsUrl),
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
    onBootProgress: vi.fn((callback: (payload: Record<string, unknown>) => void) => {
      bootProgressHandler = callback

      return () => {
        bootProgressHandler = null
      }
    }),
    // Test helper: fire a post-boot progress event through the real subscription.
    emitBootProgress(payload: Record<string, unknown>) {
      bootProgressHandler?.(payload)
    },
    onBackendExit: vi.fn((callback: (payload?: unknown) => void) => {
      backendExit = callback

      return () => {
        backendExit = null
      }
    }),
    onConnectionApplied: vi.fn((callback: () => void) => {
      connectionApplied = callback

      return () => {
        connectionApplied = null
      }
    }),
    onPowerResume: vi.fn((callback: () => void) => {
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
  handleGatewayEvent = () => undefined,
  refreshHermesConfig = async () => undefined,
  refreshSessions
}: {
  beforeConnectionSwitch?: () => void
  handleGatewayEvent?: (event: GatewayEvent) => void
  refreshHermesConfig?: (force?: boolean, shouldPublish?: () => boolean) => Promise<void>
  refreshSessions?: (shouldPublish?: () => boolean) => Promise<void>
} = {}) {
  useGatewayBoot({
    beforeConnectionSwitch,
    handleGatewayEvent,
    handleServerRequest: () => false,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig,
    refreshSessions: refreshSessions ?? (async () => undefined)
  })

  return null
}

const originalWebSocket = globalThis.WebSocket

/** The suite's `beforeEach`: one clean, fake-timed, fake-bridged window. */
export function resetGatewayBootHarness() {
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
  FakeWebSocket.readyPayload = {}
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
}

/** The suite's `afterEach`: unmount and put every global back. */
export function teardownGatewayBootHarness() {
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
  delete (document as { visibilityState?: unknown }).visibilityState
  window.localStorage.removeItem('hermes.desktop.workspace-cwd')
  $currentCwd.set('')
  $busy.set(false)
  $awaitingResponse.set(false)
}

// Let pending microtasks (awaits) AND the queued 0ms socket open/error fire.
export async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

// Drive the exponential backoff forward by its full cap so the next scheduled
// reconnect attempt actually runs (1s,2s,4s,8s,15s,15s…). Returns after the
// attempt's async work settles.
export async function advanceBackoff() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(15_000)
  })
}

// Window visibility is a document property the wake path reads directly, so the
// tests drive it the way the browser does: swap the value, dispatch the event.
export function setVisibility(state: 'hidden' | 'visible') {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state })
  document.dispatchEvent(new Event('visibilitychange'))
}

/** Parse one C1 `reconnect-episode …` telemetry line into its `key=value` fields. */
export function parseReconnectEpisode(line: string): null | Record<string, string> {
  if (!line.includes('reconnect-episode')) {
    return null
  }

  return Object.fromEntries([...line.matchAll(/([a-z_]+)=(\S+)/g)].map(match => [match[1], match[2]]))
}

export type { DesktopConnectionsRegistry }
