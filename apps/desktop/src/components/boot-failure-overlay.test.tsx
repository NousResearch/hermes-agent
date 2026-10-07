import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $desktopBoot } from '@/store/boot'
import { $notifications } from '@/store/notifications'
import { $desktopOnboarding } from '@/store/onboarding'

import { BootFailureOverlay } from './boot-failure-overlay'

// Remote-backend users hit a hard boot failure that isn't OAuth reauth (token
// auth, wrong URL, unreachable host). The recovery screen must let them fix the
// remote connection in place — the "Connection settings" action swaps the card
// to an in-line connect form — instead of stranding them (the old bug forced a
// hand-edit of connection.json).

function failBoot() {
  $desktopBoot.set({
    error: 'Could not connect to Rabbit gateway',
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
  const original = window.rabbitDesktop
  Object.defineProperty(window, 'rabbitDesktop', {
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

  return () => Object.defineProperty(window, 'rabbitDesktop', { configurable: true, value: original })
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

    const recoverySurface = screen.getByRole('dialog', { name: /Rabbit couldn't start/i })
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

  it('hides the modal on dismiss without clearing the boot error', () => {
    const { rerender } = render(<BootFailureOverlay />)
    const error = $desktopBoot.get().error

    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))

    expect(screen.queryByRole('dialog')).toBeNull()
    expect($desktopBoot.get().error).toBe(error)
    expect($desktopBoot.get().error).toBeTruthy()

    $desktopBoot.set({ ...$desktopBoot.get(), error: 'A different startup failure' })
    rerender(<BootFailureOverlay />)
    expect(screen.getByRole('dialog', { name: /Rabbit couldn't start/i })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))
    expect(screen.queryByRole('dialog')).toBeNull()

    $desktopBoot.set({ ...$desktopBoot.get(), error: null, running: false })
    rerender(<BootFailureOverlay />)
    $desktopBoot.set({ ...$desktopBoot.get(), error, running: false })
    rerender(<BootFailureOverlay />)

    expect(screen.getByRole('dialog', { name: /Rabbit couldn't start/i })).toBeTruthy()
  })

  it('dismisses on Escape and keeps the boot error latched', () => {
    render(<BootFailureOverlay />)
    const error = $desktopBoot.get().error

    fireEvent.keyDown(screen.getByRole('dialog', { name: /Rabbit couldn't start/i }), { key: 'Escape' })

    expect(screen.queryByRole('dialog')).toBeNull()
    expect($desktopBoot.get().error).toBe(error)
  })

  it('dismisses from the embedded gateway settings view', async () => {
    render(<BootFailureOverlay />)

    fireEvent.click(screen.getByRole('button', { name: /gateway settings/i }))
    expect(await screen.findByRole('dialog', { name: /gateway settings/i })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))

    expect(screen.queryByRole('dialog')).toBeNull()
    expect($desktopBoot.get().error).toBeTruthy()
  })

  it('re-shows the same error after a retry starts and fails again', () => {
    render(<BootFailureOverlay />)
    const error = $desktopBoot.get().error

    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))
    expect(screen.queryByRole('dialog')).toBeNull()

    act(() => $desktopBoot.set({ ...$desktopBoot.get(), running: true }))
    act(() => $desktopBoot.set({ ...$desktopBoot.get(), error, running: false }))

    expect(screen.getByRole('dialog', { name: /Rabbit couldn't start/i })).toBeTruthy()
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
        'This app bundles its own Rabbit runtime, but the runtime files are missing or damaged. Reinstall Rabbit Desktop to restore it.'
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
        expect(openExternal).toHaveBeenCalledWith('https://github.com/seven0070/Rabbit-/tree/main/website/docs/user-guide/desktop')
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
