// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import {
  $desktopMetricsGate,
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  recordLongFrames,
  resetDesktopMetricsForTests,
  tickDesktopMetrics
} from '@/store/desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from '@/store/desktop-metrics-test-locks'
import type * as GatewayStore from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsScopeOverride } from '@/store/settings-scope'
import type { SharedMetricsConsent } from '@/store/shared-metrics'
import { pageStorage } from '@/test/page-storage'

const requestGatewayForAgent = vi.fn()

// The real gateway store, with only the per-agent request routed to the fake backend below.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))

const { en } = await import('@/i18n/en')
const { SharedMetricsSettings } = await import('./shared-metrics-settings')

const copy = en.sharedMetrics

function backend(initial: SharedMetricsConsent) {
  let stored = initial

  requestGatewayForAgent.mockImplementation(
    async (_connection: unknown, _profile: unknown, method: string, params: Record<string, unknown> = {}) => {
      if (method === 'shared_metrics.set') {
        const enabled = params.enabled === true
        stored = { enabled, send: enabled && params.send === true, decided: true }
      }

      return stored
    }
  )
}

const setCalls = () =>
  requestGatewayForAgent.mock.calls
    .filter(call => call[2] === 'shared_metrics.set')
    .map(call => ({ enabled: call[3].enabled, send: call[3].send }))

const initialOwner = {
  connection: $connection.get(),
  profile: $activeGatewayProfile.get(),
  settings: $settingsScopeOverride.get()
}

const sourceConnection: HermesConnection = {
  baseUrl: 'http://profile.invalid',
  connectionId: 'source-server',
  isFullscreen: false,
  logs: [],
  mode: 'remote',
  nativeOverlayWidth: 0,
  profile: 'default',
  token: 'test',
  windowButtonPosition: null,
  wsUrl: 'ws://profile.invalid/ws'
}

beforeEach(() => {
  immediateMetricsLocks()
  pageStorage().clear()
  resetDesktopMetricsForTests()
  $connection.set(sourceConnection)
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  bindDesktopMetrics(null, 'source-server|default')
})

afterEach(() => {
  cleanup()
  requestGatewayForAgent.mockReset()
  resetDesktopMetricsForTests()
  $connection.set(initialOwner.connection)
  $activeGatewayProfile.set(initialOwner.profile)
  $settingsScopeOverride.set(initialOwner.settings)
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('SharedMetricsSettings', () => {
  it('is one switch: on shares (both opt-ins), off clears both', async () => {
    backend({ enabled: false, send: false, decided: true })

    render(<SharedMetricsSettings />)

    const share = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect(share.hasAttribute('disabled')).toBe(false))
    expect(screen.getAllByRole('switch')).toHaveLength(1)
    expect(share.getAttribute('aria-checked')).toBe('false')

    fireEvent.click(share)
    await waitFor(() => expect(share.getAttribute('aria-checked')).toBe('true'))

    fireEvent.click(share)
    await waitFor(() => expect(share.getAttribute('aria-checked')).toBe('false'))

    expect(setCalls()).toEqual([
      { enabled: true, send: true },
      { enabled: false, send: false }
    ])
  })

  it('reads a retired collect-only answer as not sharing', async () => {
    backend({ enabled: true, send: false, decided: true })

    render(<SharedMetricsSettings />)

    const share = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect(share.hasAttribute('disabled')).toBe(false))
    expect(share.getAttribute('aria-checked')).toBe('false')
  })

  it('keeps the newly focused connection collecting when an old connection finishes opting out', async () => {
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(new Date('2026-09-27T12:00:00Z'))
    const off = Promise.withResolvers<SharedMetricsConsent>()
    const report = vi.fn().mockResolvedValue({ recorded: true })

    requestGatewayForAgent.mockImplementation(async (_connection, _profile, method) =>
      method === 'shared_metrics.set'
        ? off.promise
        : { enabled: true, send: true, decided: true }
    )
    await setMetricsGate(captureDesktopMetricsGate, 'on')
    render(<SharedMetricsSettings />)
    const share = await screen.findByRole('switch', { name: copy.sendLabel })

    await waitFor(() => expect(share.hasAttribute('disabled')).toBe(false))
    fireEvent.click(share)
    await waitFor(() => expect(setCalls()).toEqual([{ enabled: false, send: false }]))

    await act(async () => {
      $connection.set({ ...sourceConnection, connectionId: 'target-server' })
      bindDesktopMetrics(report, 'target-server|default')
      await setMetricsGate(captureDesktopMetricsGate, 'on')
    })
    await waitFor(() => expect($desktopMetricsGate.get()).toBe('on'))
    const targetShare = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect(targetShare.hasAttribute('disabled')).toBe(false))
    recordLongFrames([120])
    await act(async () => {
      off.resolve({ enabled: false, send: false, decided: true })
      await off.promise
    })

    expect($desktopMetricsGate.get()).toBe('on')
    expect(targetShare.getAttribute('aria-checked')).toBe('true')
    vi.setSystemTime(new Date('2026-09-28T12:00:00Z'))
    tickDesktopMetrics()
    await waitFor(() => expect(report).toHaveBeenCalledWith(
      'shared_metrics.desktop_long_frames',
      expect.objectContaining({
        day: '2026-09-27',
        frames: [{ bucket: '100ms_to_250ms', count: 1 }]
      })
    ))
  })
})
