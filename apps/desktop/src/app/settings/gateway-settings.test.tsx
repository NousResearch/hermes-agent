import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionConfig, DesktopConnectionProbeResult } from '@/global'

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/profile', async () => {
  const { atom } = await vi.importActual<typeof import('nanostores')>('nanostores')

  return {
    $profiles: atom([]),
    refreshActiveProfile: vi.fn(async () => undefined)
  }
})

const MERCURY_URL = 'http://mercury2:9119'

const getConnectionConfig = vi.fn<() => Promise<DesktopConnectionConfig>>()
const oauthLoginConnectionConfig = vi.fn<() => Promise<{ baseUrl: string; connected: boolean; ok: boolean }>>()
const probeConnectionConfig = vi.fn<() => Promise<DesktopConnectionProbeResult>>()
const saveConnectionConfig = vi.fn<(payload: unknown) => Promise<DesktopConnectionConfig>>()

function connectionConfig(overrides: Partial<DesktopConnectionConfig> = {}): DesktopConnectionConfig {
  return {
    envOverride: true,
    mode: 'remote',
    profile: null,
    remoteAuthMode: 'oauth',
    remoteOauthConnected: false,
    remoteTokenPreview: null,
    remoteTokenSet: false,
    remoteUrl: MERCURY_URL,
    ...overrides
  }
}

function probeResult(overrides: Partial<DesktopConnectionProbeResult> = {}): DesktopConnectionProbeResult {
  return {
    authMode: 'oauth',
    baseUrl: MERCURY_URL,
    error: null,
    providers: [{ displayName: 'Basic', name: 'basic', supportsPassword: true }],
    reachable: true,
    version: null,
    ...overrides
  }
}

beforeEach(() => {
  getConnectionConfig.mockResolvedValue(connectionConfig())
  oauthLoginConnectionConfig.mockResolvedValue({ baseUrl: MERCURY_URL, connected: true, ok: true })
  probeConnectionConfig.mockResolvedValue(probeResult())
  saveConnectionConfig.mockResolvedValue(connectionConfig())

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      getConnectionConfig,
      oauthLoginConnectionConfig,
      probeConnectionConfig,
      saveConnectionConfig
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  delete (window as Partial<Window>).hermesDesktop
})

async function renderGatewaySettings() {
  const { GatewaySettings } = await import('./gateway-settings')

  return render(<GatewaySettings />)
}

describe('GatewaySettings', () => {
  it('persists the env-provided OAuth URL before opening login', async () => {
    await renderGatewaySettings()

    expect(await screen.findByDisplayValue(MERCURY_URL)).toBeTruthy()
    await waitFor(() => expect(probeConnectionConfig).toHaveBeenCalledWith(MERCURY_URL))

    fireEvent.click(await screen.findByRole('button', { name: 'Sign in' }))

    await waitFor(() =>
      expect(saveConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        profile: undefined,
        remoteAuthMode: 'oauth',
        remoteUrl: MERCURY_URL
      })
    )
    expect(oauthLoginConnectionConfig).toHaveBeenCalledWith(MERCURY_URL)
  })

  it('allows saving an env-provided OAuth gateway for future app launches', async () => {
    getConnectionConfig.mockResolvedValue(connectionConfig({ remoteOauthConnected: true }))
    saveConnectionConfig.mockResolvedValue(connectionConfig({ remoteOauthConnected: true }))

    await renderGatewaySettings()

    fireEvent.click(await screen.findByRole('button', { name: 'Save for next restart' }))

    await waitFor(() =>
      expect(saveConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        profile: undefined,
        remoteAuthMode: 'oauth',
        remoteToken: undefined,
        remoteUrl: MERCURY_URL
      })
    )
  })
})
