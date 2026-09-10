import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { $desktopOnboarding } from '@/store/onboarding'

import { BootFailureOverlay } from './boot-failure-overlay'

// Remote-backend users hit a hard boot failure that isn't OAuth reauth (token
// auth, wrong URL, unreachable host). The recovery screen must let them fix the
// remote connection in place — the "Connection settings" action swaps the card
// to an in-line connect form — instead of stranding them (the old bug forced a
// hand-edit of connection.json).

function failBoot() {
  $desktopBoot.set({
    error: 'Could not connect to Hermes gateway',
    fakeMode: false,
    message: 'boot failed',
    phase: 'renderer.error',
    progress: 40,
    running: false,
    timestamp: Date.now(),
    visible: true
  })
}

function stubDesktop(config: Record<string, unknown>, overrides: Record<string, unknown> = {}) {
  const original = window.hermesDesktop
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getRecentLogs: async () => ({ lines: [] }), getConnectionConfig: async () => config, ...overrides }
  })

  return () => Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: original })
}

const remoteToken = {
  envOverride: false,
  mode: 'remote',
  profile: null,
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: true,
  remoteUrl: 'http://100.116.104.53:9191',
  cloudOrg: ''
}

beforeEach(() => {
  $desktopOnboarding.set({
    configured: true,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  })
  failBoot()
})

afterEach(cleanup)

describe('BootFailureOverlay', () => {
  it('swaps to the in-place gateway settings view (no route nav) and back', async () => {
    render(<BootFailureOverlay />)

    fireEvent.click(screen.getByRole('button', { name: /gateway settings/i }))
    // Recovery actions give way to the embedded panel (behind a Back control).
    expect(await screen.findByRole('button', { name: /back/i })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /retry/i })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /back/i }))
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /back/i })).toBeNull()
  })

  it('drops local-only Repair and Use-local-gateway on a local failure', () => {
    render(<BootFailureOverlay />)
    // No connection config stub → treated as a local failure.
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /repair/i })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /use local gateway/i })).toBeNull()
  })

  it('leads with Gateway settings and drops Repair for a remote (token) failure', async () => {
    const restore = stubDesktop(remoteToken)

    try {
      render(<BootFailureOverlay />)
      await waitFor(() => expect(screen.queryByRole('button', { name: /repair/i })).toBeNull())
      expect(screen.getByRole('button', { name: /gateway settings/i })).toBeTruthy()
      expect(screen.getByRole('button', { name: /use local gateway/i })).toBeTruthy()
    } finally {
      restore()
    }
  })

  it('opens gateway settings with a partial persisted remote config', async () => {
    const restore = stubDesktop({ mode: 'remote', remoteAuthMode: undefined, remoteUrl: undefined })

    try {
      render(<BootFailureOverlay />)
      fireEvent.click(screen.getByRole('button', { name: /gateway settings/i }))

      expect(await screen.findByRole('button', { name: /back/i })).toBeTruthy()
      expect(screen.queryByRole('button', { name: /retry/i })).toBeNull()
    } finally {
      restore()
    }
  })

  it('clears and signs in only the failed gateway once', async () => {
    const gatewayUrl = 'http://100.116.104.53:9191'
    const logout = vi.fn().mockResolvedValue({ ok: true, connected: false })
    const login = vi.fn().mockResolvedValue({ ok: true, connected: false })

    const restore = stubDesktop(
      {
        ...remoteToken,
        remoteAuthMode: 'oauth',
        remoteOauthConnected: false,
        remoteTokenSet: false,
        remoteUrl: gatewayUrl
      },
      {
        oauthLoginConnectionConfig: login,
        oauthLogoutConnectionConfig: logout,
        probeConnectionConfig: vi.fn().mockResolvedValue({ providers: [{ id: 'basic', type: 'password' }] })
      }
    )

    try {
      render(<BootFailureOverlay />)
      fireEvent.click(await screen.findByRole('button', { name: /sign out & sign in/i }))

      await waitFor(() => expect(login).toHaveBeenCalledWith(gatewayUrl))
      expect(logout).toHaveBeenCalledTimes(1)
      expect(logout).toHaveBeenCalledWith(gatewayUrl)
      expect(login).toHaveBeenCalledTimes(1)
    } finally {
      restore()
    }
  })

  it('recovers a cloud connection through the portal cascade instead of native OAuth', async () => {
    const gatewayUrl = 'https://agent-1.agents.nousresearch.com'
    const logout = vi.fn().mockResolvedValue({ ok: true, connected: false })
    const nativeLogin = vi.fn().mockResolvedValue({ ok: true, connected: false })
    const cloudStatus = vi.fn().mockResolvedValue({ portalBaseUrl: 'https://portal.nousresearch.com', signedIn: false })

    const cloudLogin = vi.fn().mockResolvedValue({
      ok: true,
      portalBaseUrl: 'https://portal.nousresearch.com',
      signedIn: true
    })

    const cloudAgentSignIn = vi.fn().mockResolvedValue({ baseUrl: gatewayUrl, connected: false })

    const restore = stubDesktop(
      {
        ...remoteToken,
        mode: 'cloud',
        remoteAuthMode: 'oauth',
        remoteOauthConnected: false,
        remoteTokenSet: false,
        remoteUrl: gatewayUrl
      },
      {
        cloud: { status: cloudStatus, login: cloudLogin, agentSignIn: cloudAgentSignIn },
        oauthLoginConnectionConfig: nativeLogin,
        oauthLogoutConnectionConfig: logout,
        probeConnectionConfig: vi.fn().mockResolvedValue({ providers: [{ id: 'nous', type: 'oauth' }] })
      }
    )

    try {
      render(<BootFailureOverlay />)
      fireEvent.click(await screen.findByRole('button', { name: /sign in/i }))

      await waitFor(() => expect(cloudAgentSignIn).toHaveBeenCalledWith(gatewayUrl))
      expect(logout).toHaveBeenCalledWith(gatewayUrl)
      expect(cloudStatus).toHaveBeenCalledTimes(1)
      expect(cloudLogin).toHaveBeenCalledTimes(1)
      expect(nativeLogin).not.toHaveBeenCalled()
    } finally {
      restore()
    }
  })

  it('shows the Nous Cloud down recovery when the backend flags isCloudBackendDown', async () => {
    const restore = stubDesktop(remoteToken)
    $desktopBoot.set({
      error: 'Nous Cloud agent ares-3009.agents.nousresearch.com is down (HTTP 503: server-side fault).',
      fakeMode: false,
      isCloudBackendDown: true,
      message: 'boot failed',
      phase: 'renderer.error',
      progress: 40,
      running: false,
      statusCode: 503,
      timestamp: Date.now(),
      visible: true
    })

    try {
      render(<BootFailureOverlay />)
      // Cloud-specific title + actionable recovery instead of the generic
      // remote-failure copy.
      expect(await screen.findByText(/Nous Cloud agent is down/i)).toBeTruthy()
      // Portal and Discord are dedicated action buttons (localized labels
      // can't drift the URLs, which live in code).
      expect(screen.getByRole('button', { name: /check portal status/i })).toBeTruthy()
      expect(screen.getByRole('button', { name: /get help on discord/i })).toBeTruthy()
      // Cloud-down is a remote failure: local-only Repair is dropped; the
      // actionable paths are Gateway settings + Use local gateway.
      expect(screen.queryByRole('button', { name: /repair/i })).toBeNull()
      expect(screen.getByRole('button', { name: /gateway settings/i })).toBeTruthy()
      expect(screen.getByRole('button', { name: /use local gateway/i })).toBeTruthy()
      // The electron-built error message (portal / local mode / Discord) is
      // still surfaced in the error box.
      expect(screen.getByText(/ares-3009\.agents\.nousresearch\.com/i)).toBeTruthy()
    } finally {
      restore()
    }
  })

  it('ipc-bridge failure (#92927): honest repair copy, no bridge-dependent dead-end actions', () => {
    $desktopBoot.set({
      error: 'Desktop IPC bridge is unavailable.',
      errorKind: 'ipc-bridge',
      fakeMode: false,
      message: 'boot failed',
      phase: 'renderer.error',
      progress: 6,
      running: false,
      timestamp: Date.now(),
      visible: true
    })

    render(<BootFailureOverlay />)

    // The ipc-specific title and repair guidance replace the generic
    // "background gateway didn't come up" copy.
    expect(screen.getByRole('heading', { name: /Desktop IPC bridge is unavailable/i })).toBeTruthy()
    expect(screen.getByText(/hermes desktop --force-build/i)).toBeTruthy()
    // Reload is the only action that works without the bridge; Repair,
    // Gateway settings and Open logs all round-trip through
    // window.hermesDesktop and would be silent no-ops here.
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /repair/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /gateway settings/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /open logs/i })).toBeNull()
  })

  it('rpc-probe failure (#92927): keys the torn-install hint on the stable errorCode', () => {
    $desktopBoot.set({
      error: 'Local Hermes backend is reachable but its JSON-RPC gateway did not answer (Timed out after 8000ms waiting for a JSON-RPC reply to "session.list".).',
      errorCode: 'gateway.rpc-probe-failed',
      fakeMode: false,
      message: 'boot failed',
      phase: 'renderer.error',
      progress: 40,
      running: false,
      timestamp: Date.now(),
      visible: true
    })

    render(<BootFailureOverlay />)

    // The localized torn-install guidance replaces the generic repair hint,
    // keyed on the stable code (not string-matching the message).
    expect(screen.getByText(/repair it from a terminal/i)).toBeTruthy()
    expect(screen.getByText(/hermes desktop --force-build/i)).toBeTruthy()
    // The bridge is alive here, so the full local recovery set stays.
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /repair install/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /gateway settings/i })).toBeTruthy()
  })

  it('rpc-probe unavailable (#92927): keys the capability hint on the stable errorCode', () => {
    $desktopBoot.set({
      error: 'Local Hermes backend is reachable but this runtime cannot open a WebSocket to verify its JSON-RPC gateway.',
      errorCode: 'gateway.rpc-unavailable',
      fakeMode: false,
      message: 'boot failed',
      phase: 'renderer.error',
      progress: 40,
      running: false,
      timestamp: Date.now(),
      visible: true
    })

    render(<BootFailureOverlay />)

    // A capability problem, not a broken install: the hint says so, and the
    // torn-install repair commands must NOT appear.
    expect(screen.getByText(/missing WebSocket support/i)).toBeTruthy()
    expect(screen.queryByText(/hermes desktop --force-build/i)).toBeNull()
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
  })

  it('a generic local failure without an errorCode keeps the generic repair hint', () => {
    render(<BootFailureOverlay />)

    expect(screen.getByText(/re-runs the installer/i)).toBeTruthy()
    expect(screen.queryByText(/repair it from a terminal/i)).toBeNull()
    expect(screen.queryByText(/missing WebSocket support/i)).toBeNull()
  })
})
