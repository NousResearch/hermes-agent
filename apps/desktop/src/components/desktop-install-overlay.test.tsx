import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopBootstrapState, DesktopConnectionProbeResult } from '@/global'

import { DesktopInstallOverlay } from './desktop-install-overlay'

const unsupportedState: DesktopBootstrapState = {
  active: false,
  completedAt: null,
  error: null,
  log: [],
  manifest: null,
  stages: {},
  startedAt: null,
  unsupportedPlatform: {
    activeRoot: '/opt/hermes',
    docsUrl: 'https://example.com/install',
    installCommand: 'curl -fsSL https://example.com/install.sh | sh',
    platform: 'darwin'
  }
}

function installOverlayDesktop(probe: DesktopConnectionProbeResult) {
  return {
    applyConnectionConfig: vi.fn().mockResolvedValue({}),
    getBootstrapState: vi.fn().mockResolvedValue(unsupportedState),
    oauthLoginConnectionConfig: vi.fn().mockResolvedValue({ baseUrl: probe.baseUrl, connected: true, ok: true }),
    onBootstrapEvent: vi.fn().mockReturnValue(() => undefined),
    openExternal: vi.fn().mockResolvedValue(undefined),
    probeConnectionConfig: vi.fn().mockResolvedValue(probe),
    saveConnectionConfig: vi.fn().mockResolvedValue({}),
    testConnectionConfig: vi.fn().mockResolvedValue({ baseUrl: probe.baseUrl, ok: true, version: '0.15.1' })
  }
}

describe('DesktopInstallOverlay unsupported-platform remote onboarding', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('shows the remote-connect path alongside the manual install command', async () => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = installOverlayDesktop({
      authMode: 'token',
      baseUrl: 'https://gw.example.com',
      error: null,
      providers: [],
      reachable: true,
      version: '0.15.1'
    })

    render(<DesktopInstallOverlay />)

    expect(await screen.findByText('Hermes needs a one-time install')).toBeTruthy()
    expect(screen.getByText('Connect to existing Hermes')).toBeTruthy()
    expect(screen.getByLabelText('Remote dashboard URL')).toBeTruthy()
  })

  it('probes token auth remotes and applies the saved remote connection', async () => {
    const desktop = installOverlayDesktop({
      authMode: 'token',
      baseUrl: 'https://gw.example.com',
      error: null,
      providers: [],
      reachable: true,
      version: '0.15.1'
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<DesktopInstallOverlay />)
    await screen.findByText('Connect to existing Hermes')

    fireEvent.change(screen.getByLabelText('Remote dashboard URL'), {
      target: { value: 'https://gw.example.com' }
    })

    await vi.advanceTimersByTimeAsync(450)
    await waitFor(() => expect(desktop.probeConnectionConfig).toHaveBeenCalledWith('https://gw.example.com'))

    fireEvent.change(screen.getByLabelText('Session token'), {
      target: { value: 'secret-token' }
    })

    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }))
    await waitFor(() =>
      expect(desktop.testConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        remoteAuthMode: 'token',
        remoteToken: 'secret-token',
        remoteUrl: 'https://gw.example.com'
      })
    )

    fireEvent.click(screen.getByRole('button', { name: 'Use this Hermes instance' }))
    await waitFor(() =>
      expect(desktop.applyConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        remoteAuthMode: 'token',
        remoteToken: 'secret-token',
        remoteUrl: 'https://gw.example.com'
      })
    )
  })

  it('shows OAuth sign-in for gated remotes before allowing connect', async () => {
    const desktop = installOverlayDesktop({
      authMode: 'oauth',
      baseUrl: 'https://gw.example.com',
      error: null,
      providers: [{ displayName: 'Nous Portal', name: 'nous', supportsPassword: false }],
      reachable: true,
      version: '0.15.1'
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<DesktopInstallOverlay />)
    await screen.findByText('Connect to existing Hermes')

    fireEvent.change(screen.getByLabelText('Remote dashboard URL'), {
      target: { value: 'https://gw.example.com' }
    })

    await vi.advanceTimersByTimeAsync(450)
    expect(await screen.findByRole('button', { name: 'Sign in with Nous Portal' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Sign in with Nous Portal' }))
    await waitFor(() =>
      expect(desktop.saveConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        remoteAuthMode: 'oauth',
        remoteUrl: 'https://gw.example.com'
      })
    )
    await waitFor(() => expect(desktop.oauthLoginConnectionConfig).toHaveBeenCalledWith('https://gw.example.com'))

    fireEvent.click(screen.getByRole('button', { name: 'Use this Hermes instance' }))
    await waitFor(() =>
      expect(desktop.applyConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        remoteAuthMode: 'oauth',
        remoteToken: undefined,
        remoteUrl: 'https://gw.example.com'
      })
    )
  })
})
