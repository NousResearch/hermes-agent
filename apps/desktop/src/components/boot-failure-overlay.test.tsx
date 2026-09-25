import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $desktopBoot } from '@/store/boot'
import { $notifications } from '@/store/notifications'
import { $desktopOnboarding } from '@/store/onboarding'

import { BootFailureOverlay } from './boot-failure-overlay'

const { notifyMock } = vi.hoisted(() => ({ notifyMock: vi.fn() }))

vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<any>()),
  notify: notifyMock
}))

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
    value: {
      getRecentLogs: async () => ({ lines: [] }),
      getConnectionConfig: async () => config,
      getBootstrapState: async () => ({
        active: false,
        manifest: null,
        stages: {},
        error: null,
        log: [],
        startedAt: null,
        completedAt: null,
        setupChoice: null,
        unsupportedPlatform: null,
        bundled: false
      }),
      ...overrides
    }
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
  it('keeps keyboard focus inside the recovery surface', () => {
    render(
      <>
        <button type="button">Background action</button>
        <BootFailureOverlay />
      </>
    )

    const recoverySurface = screen.getByRole('dialog', { name: /Hermes couldn't start/i })
    const retry = screen.getByRole('button', { name: /retry/i })
    const backgroundAction = screen.getByText(/background action/i)

    retry.focus()
    backgroundAction.focus()

    expect(recoverySurface.getAttribute('aria-modal')).toBe('true')
    expect(recoverySurface.contains(globalThis.document.activeElement)).toBe(true)
  })

  it('swaps to the in-place gateway settings view (no route nav) and back', async () => {
    render(<BootFailureOverlay />)

    fireEvent.click(screen.getByRole('button', { name: /gateway settings/i }))
    // Recovery actions give way to the embedded panel (behind a Back control).
    expect(await screen.findByRole('button', { name: /back/i })).toBeTruthy()
    expect(screen.getByRole('dialog', { name: /gateway settings/i }).getAttribute('aria-modal')).toBe('true')
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
      // The ladder owns the logout: exactly one drop of this gateway's cookies.
      expect(logout).toHaveBeenCalledExactlyOnceWith(gatewayUrl)
      expect(cloudStatus).toHaveBeenCalledTimes(1)
      expect(cloudLogin).toHaveBeenCalledTimes(1)
      expect(nativeLogin).not.toHaveBeenCalled()
    } finally {
      restore()
    }
  })

  const cloudReauthConfig = {
    ...remoteToken,
    mode: 'cloud',
    remoteAuthMode: 'oauth',
    remoteOauthConnected: false,
    remoteTokenSet: false,
    remoteUrl: 'https://agent-1.agents.nousresearch.com'
  }

  const cloudReauthDeps = (cloud: Record<string, unknown>) => ({
    cloud: {
      status: vi.fn().mockResolvedValue({ portalBaseUrl: 'https://portal.nousresearch.com', signedIn: false }),
      ...cloud
    },
    oauthLogoutConnectionConfig: vi.fn().mockResolvedValue({ ok: true, connected: false }),
    probeConnectionConfig: vi.fn().mockResolvedValue({ providers: [{ id: 'nous', type: 'oauth' }] })
  })

  it('a denied or cancelled cloud browser sign-in is quiet: no warning toast, no agent sign-in', async () => {
    const cloudAgentSignIn = vi.fn()

    const restore = stubDesktop(
      cloudReauthConfig,
      cloudReauthDeps({
        login: vi.fn().mockResolvedValue({ ok: false, signedIn: false, cancelled: true }),
        agentSignIn: cloudAgentSignIn
      })
    )

    try {
      notifyMock.mockClear()
      render(<BootFailureOverlay />)
      fireEvent.click(await screen.findByRole('button', { name: /sign in/i }))

      await waitFor(() => expect(screen.getByRole('button', { name: /sign in/i }).hasAttribute('disabled')).toBe(false))
      expect(notifyMock).not.toHaveBeenCalled()
      expect(cloudAgentSignIn).not.toHaveBeenCalled()
    } finally {
      restore()
    }
  })

  it('a cloud browser sign-in that did not complete gets browser copy, not "login window closed"', async () => {
    const restore = stubDesktop(
      cloudReauthConfig,
      cloudReauthDeps({ login: vi.fn().mockResolvedValue({ ok: false, signedIn: false }), agentSignIn: vi.fn() })
    )

    try {
      notifyMock.mockClear()
      render(<BootFailureOverlay />)
      fireEvent.click(await screen.findByRole('button', { name: /sign in/i }))

      await waitFor(() =>
        expect(notifyMock).toHaveBeenCalledWith(
          expect.objectContaining({ kind: 'warning', message: 'Sign-in was not completed in the browser.' })
        )
      )
      expect(notifyMock).not.toHaveBeenCalledWith(
        expect.objectContaining({ message: 'The login window closed before authentication finished.' })
      )
    } finally {
      restore()
    }
  })

  it('while the cloud browser sign-in is pending, offers Copy sign-in link (once a URL exists) and Cancel sign-in', async () => {
    let finishLogin!: (value: unknown) => void
    const url = 'https://portal.nousresearch.com/oauth/authorize?client_id=hermes-desktop'
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })

    const cancelLogin = vi.fn(async () => {
      finishLogin({ ok: false, signedIn: false, cancelled: true })

      return { cancelled: true }
    })

    const restore = stubDesktop(
      cloudReauthConfig,
      cloudReauthDeps({
        login: vi.fn(() => new Promise(resolve => (finishLogin = resolve))),
        cancelLogin,
        loginUrl: vi.fn().mockResolvedValue({ url }),
        agentSignIn: vi.fn()
      })
    )

    try {
      notifyMock.mockClear()
      render(<BootFailureOverlay />)
      fireEvent.click(await screen.findByRole('button', { name: /sign in/i }))

      const status = await screen.findByRole('status')
      const copyLink = await screen.findByRole('button', { name: 'Copy sign-in link' })
      await waitFor(() => expect(copyLink.hasAttribute('disabled')).toBe(false))
      fireEvent.click(copyLink)
      await waitFor(() => expect(writeText).toHaveBeenCalledWith(url))
      expect(status.getAttribute('aria-live')).toBe('polite')

      fireEvent.click(screen.getByRole('button', { name: 'Cancel sign-in' }))
      await waitFor(() => expect(cancelLogin).toHaveBeenCalledTimes(1))
      await waitFor(() => expect(screen.queryByRole('button', { name: 'Cancel sign-in' })).toBeNull())
      expect(notifyMock).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'warning' }))
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

  const bundledState = {
    active: false,
    manifest: null,
    stages: {},
    error: null,
    log: [],
    startedAt: null,
    completedAt: null,
    setupChoice: null,
    unsupportedPlatform: null,
    bundled: true
  }

  it('offers "Reinstall the app" on a bundled install only when the payload itself is damaged', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    const restore = stubDesktop({ mode: 'local' }, { getBootstrapState: async () => bundledState, openExternal })
    $desktopBoot.set({
      ...$desktopBoot.get(),
      error:
        'This app bundles its own Hermes runtime, but the runtime files are missing or damaged. Reinstall Hermes Desktop to restore it.'
    })

    try {
      render(<BootFailureOverlay />)

      // The bundled artifact has no installer to repair with — the action is
      // a docs link, and the hint says reinstall instead of re-run installer.
      expect(await screen.findByRole('button', { name: /reinstall the app/i })).toBeTruthy()
      expect(screen.queryByRole('button', { name: /repair install/i })).toBeNull()
      expect(screen.getByText(/reinstall the app to restore/i)).toBeTruthy()

      fireEvent.click(screen.getByRole('button', { name: /reinstall the app/i }))
      await waitFor(() =>
        expect(openExternal).toHaveBeenCalledWith('https://hermes-agent.nousresearch.com/docs/user-guide/desktop')
      )
    } finally {
      restore()
    }
  })

  it('a bundled install with an unrelated failure gets neither Repair nor Reinstall', async () => {
    const restore = stubDesktop({ mode: 'local' }, { getBootstrapState: async () => bundledState })
    $desktopBoot.set({ ...$desktopBoot.get(), error: 'listen EADDRINUSE: address already in use 127.0.0.1:8642' })

    try {
      render(<BootFailureOverlay />)

      expect(await screen.findByRole('button', { name: /retry/i })).toBeTruthy()
      // Wait for the bundled snapshot to land before asserting the negatives.
      await waitFor(() => expect(screen.queryByRole('button', { name: /repair install/i })).toBeNull())
      expect(screen.queryByRole('button', { name: /reinstall the app/i })).toBeNull()
      expect(screen.queryByText(/reinstall the app to restore/i)).toBeNull()
    } finally {
      restore()
    }
  })

  it.each(['refused', 'thrown', 'unavailable'])('preserves a %s repair failure without reloading', async failure => {
    const reload = vi.fn()
    const originalLocation = Object.getOwnPropertyDescriptor(window, 'location')!
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...window.location, reload }
    })

    const repair =
      failure === 'unavailable'
        ? undefined
        : vi.fn(async () => {
            if (failure === 'thrown') {
              throw new Error('installer permission denied')
            }

            return { ok: false, error: 'bundled-immutable' }
          })

    const restore = stubDesktop({ mode: 'local' }, { repairBootstrap: repair })

    try {
      render(<BootFailureOverlay />)
      fireEvent.click(await screen.findByRole('button', { name: /repair install/i }))

      const message =
        failure === 'thrown'
          ? 'installer permission denied'
          : failure === 'refused'
            ? en.boot.failure.bundledReinstallHint
            : en.boot.errors.ipcBridgeUnavailable

      await waitFor(() =>
        expect($notifications.get()).toEqual(
          expect.arrayContaining([expect.objectContaining({ kind: 'error', message })])
        )
      )
      expect(reload).not.toHaveBeenCalled()
      expect(screen.getByRole('button', { name: /repair install/i }).hasAttribute('disabled')).toBe(false)
    } finally {
      restore()
      Object.defineProperty(window, 'location', originalLocation)
      $notifications.set([])
    }
  })

  it('reloads after an accepted repair', async () => {
    const reload = vi.fn()
    const originalLocation = Object.getOwnPropertyDescriptor(window, 'location')!
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...window.location, reload }
    })

    const restore = stubDesktop({ mode: 'local' }, { repairBootstrap: vi.fn().mockResolvedValue({ ok: true }) })

    try {
      render(<BootFailureOverlay />)
      fireEvent.click(await screen.findByRole('button', { name: /repair install/i }))

      await waitFor(() => expect(reload).toHaveBeenCalled())
      expect($notifications.get().some(n => n.kind === 'error')).toBe(false)
    } finally {
      restore()
      Object.defineProperty(window, 'location', originalLocation)
      $notifications.set([])
    }
  })
})
