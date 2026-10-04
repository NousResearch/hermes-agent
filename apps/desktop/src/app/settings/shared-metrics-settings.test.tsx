import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsScopeOverride, setSettingsScope } from '@/store/settings-scope'

import { SharedMetricsSettings } from './shared-metrics-settings'

const request = vi.hoisted(() => vi.fn())
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: request
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $connection.set({
    connectionId: 'local',
    mode: 'local',
    baseUrl: 'http://localhost:4242',
    wsUrl: 'ws://localhost:4242',
    token: 'test',
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  })
  request.mockImplementation(async (_connection, _profile, method, params) =>
    method === 'shared_metrics.set'
      ? { enabled: params.enabled, send: params.enabled && params.send, decided: true }
      : { enabled: false, send: false, decided: true }
  )
})

afterEach(() => {
  cleanup()
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $connection.set(null)
  vi.clearAllMocks()
})

it.each(['default', 'work', 'custom'])('reads and saves settings in the %s owning route', async profile => {
  $activeGatewayProfile.set(profile)
  render(<SharedMetricsSettings />)
  const toggle = await screen.findByRole('switch', { name: en.sharedMetrics.collectLabel })
  await waitFor(() => expect(toggle.hasAttribute('disabled')).toBe(false))
  const requestProfile = profile === 'custom' ? undefined : profile
  expect(request).toHaveBeenCalledWith(
    'local',
    profile,
    'shared_metrics.status',
    { profile: requestProfile },
    undefined,
    undefined,
    { spawnPriority: 'foreground' }
  )
  fireEvent.click(toggle)
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith(
      'local',
      profile,
      'shared_metrics.set',
      { enabled: true, send: false, first_run: false, profile: requestProfile },
      undefined,
      undefined,
      { spawnPriority: 'foreground' }
    )
  )
  const params = request.mock.calls.find(call => call[2] === 'shared_metrics.set')?.[3]
  expect(JSON.parse(JSON.stringify(params))).toEqual({
    enabled: true,
    send: false,
    first_run: false,
    ...(profile === 'custom' ? {} : { profile })
  })
})

it('keeps a named settings override explicit and re-reads on scope changes', async () => {
  $activeGatewayProfile.set('work')
  setSettingsScope('research')
  render(<SharedMetricsSettings />)
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith(
      'local',
      'research',
      'shared_metrics.status',
      { profile: 'research' },
      undefined,
      undefined,
      { spawnPriority: 'foreground' }
    )
  )
  act(() => setSettingsScope('work'))
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith(
      'local',
      'work',
      'shared_metrics.status',
      { profile: 'work' },
      undefined,
      undefined,
      { spawnPriority: 'foreground' }
    )
  )
})

it('an unresolved named target stays unavailable and cannot write ambient consent', async () => {
  setSettingsScope('missing')
  request.mockRejectedValue(new Error('Profile target unavailable (4064)'))
  render(<SharedMetricsSettings />)
  await screen.findByText(en.sharedMetrics.unavailable)
  expect(screen.getAllByRole('switch').every(toggle => toggle.hasAttribute('disabled'))).toBe(true)
  expect(request.mock.calls.every(call => call[3].profile === 'missing')).toBe(true)
  expect(request.mock.calls.some(call => call[2] === 'shared_metrics.set')).toBe(false)
})
