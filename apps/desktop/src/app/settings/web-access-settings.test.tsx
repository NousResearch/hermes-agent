const platform = vi.hoisted(() => ({ browserHosted: false }))

vi.mock('@/lib/platform', async (importOriginal): Promise<object> => ({
  ...(await importOriginal<object>()),
  isBrowserHostedDesktop: (): boolean => platform.browserHosted
}))
vi.mock('@/store/profile', async (): Promise<object> => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayProfile: atom<string>('default'),
    $profiles: atom<Array<{ name: string; is_default?: boolean }>>([]),
    normalizeProfileKey: (profile: string | null): string => profile || 'default'
  }
})
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getWebAccessStatus: vi.fn(),
  restartWebAccess: vi.fn(),
  setApiRequestProfile: vi.fn(),
  startWebAccess: vi.fn(),
  stopWebAccess: vi.fn()
}))

import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { WebAccessStatus } from '@/api/webapp-access'
import * as hermes from '@/hermes'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $notifications, clearNotifications } from '@/store/notifications'

import { WebAccessSettings } from './web-access-settings'

const mocked = vi.mocked(hermes)

const OFF: WebAccessStatus = {
  enabled: false,
  gateway: { reason: null, state: 'off', url: null },
  lan_address: '192.168.1.50',
  mode: null,
  nous_signed_in: true,
  port: 9120,
  public_url: ''
}

const SERVING: WebAccessStatus = {
  ...OFF,
  enabled: true,
  gateway: { reason: null, state: 'serving', url: 'http://192.168.1.50:9120' },
  mode: 'lan'
}

/** Turned off and saved, while the gateway that has not restarted yet still serves the web app. */
const OLD_GATEWAY: WebAccessStatus = { ...OFF, gateway: SERVING.gateway }

function renderPane(status: WebAccessStatus) {
  mocked.getWebAccessStatus.mockResolvedValue(status)

  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <WebAccessSettings />
      </I18nProvider>
    </QueryClientProvider>
  )
}

beforeEach(() => queryClient.clear())

afterEach(() => {
  cleanup()
  clearNotifications()
  platform.browserHosted = false
  vi.clearAllMocks()
  // A test that queued status answers it did not use must not hand them to the next one.
  mocked.getWebAccessStatus.mockReset()
})

describe('WebAccessSettings', () => {
  it('offers Start only with a Nous account to sign in with', async () => {
    const signedOut = renderPane({ ...OFF, nous_signed_in: false })

    expect((await screen.findByRole('button', { name: 'Start' })).hasAttribute('disabled')).toBe(true)
    signedOut.unmount()
    queryClient.clear()
    renderPane(OFF)

    expect((await screen.findByRole('button', { name: 'Start' })).hasAttribute('disabled')).toBe(false)
  })

  it('says why the gateway cannot serve a turned-on web app', async () => {
    renderPane({
      ...OFF,
      enabled: true,
      gateway: { reason: 'port 9120 is in use', state: 'unavailable', url: null },
      mode: 'lan'
    })

    expect(await screen.findByText(/port 9120 is in use/)).toBeTruthy()
  })

  it('never reports a web app the gateway still serves as stopped', async () => {
    // Stop saved `enabled: false`, but the gateway refused to restart: it still serves the web app.
    renderPane({ ...OFF, gateway: { reason: null, state: 'serving', url: 'http://127.0.0.1:9120' } })

    expect(await screen.findByText('Restart needed')).toBeTruthy()
    expect(screen.queryByText('Stopped')).toBeNull()
  })

  it('shows a web app served on this computer only as local, never as your own URL', async () => {
    // `webapp.enabled: true` on loopback (the `hermes webapp` setup): reachable from this computer only.
    renderPane({
      ...OFF,
      enabled: true,
      gateway: { reason: null, state: 'serving', url: 'http://127.0.0.1:9120' },
      mode: 'local'
    })

    expect(await screen.findByRole('link', { name: /http:\/\/127\.0\.0\.1:9120/ })).toBeTruthy()
    expect(screen.getByText(/Only this computer can open it/)).toBeTruthy()
    expect(screen.queryByText('My own URL')).toBeNull()
    expect(screen.queryByText(/192\.168\.1\.50/)).toBeNull()
    expect(screen.getByRole('button', { name: 'Stop' })).toBeTruthy()
  })

  it('restarts the gateway for a saved change without starting web access again', async () => {
    // Saved `enabled: true`, but the gateway has not restarted onto it yet (a chat refused it).
    const pending: WebAccessStatus = { ...OFF, enabled: true, mode: 'lan' }
    mocked.restartWebAccess.mockResolvedValue({ ...pending, restart_error: 'busy', restart_started: false })
    renderPane(pending)

    fireEvent.click(await screen.findByRole('button', { name: 'Restart gateway' }))

    await waitFor(() => expect(mocked.restartWebAccess).toHaveBeenCalledTimes(1))
    expect(mocked.startWebAccess).not.toHaveBeenCalled()
    expect(mocked.stopWebAccess).not.toHaveBeenCalled()
  })

  it('says plainly that a change was saved when a running chat holds off the restart', async () => {
    const refusal =
      'Refusing to restart the gateway: 1 live session(s) would be interrupted.\n' +
      '    POST /api/gateway/restart?interrupt=true   # interrupts 1 live session(s)'

    mocked.startWebAccess.mockResolvedValue({
      ...OFF,
      enabled: true,
      mode: 'lan',
      restart_busy: true,
      restart_error: refusal,
      restart_started: false
    })
    renderPane(OFF)

    fireEvent.click(await screen.findByRole('button', { name: 'Start' }))

    await waitFor(() => expect($notifications.get()).toHaveLength(1))
    const [toast] = $notifications.get()
    expect(toast.kind).toBe('warning')
    expect(toast.title).toBe('Restart needed')
    expect(toast.message).toMatch(/^Saved\. A chat is running/)
    expect(JSON.stringify(toast)).not.toMatch(/Refusing|interrupt|Couldn't change/)
  })

  it('shows nothing and asks nothing where the settings nav hides it', async () => {
    // `?tab=web-access` in the Webapp: the nav has no Web Access entry there.
    platform.browserHosted = true
    renderPane(OFF)

    await new Promise(resolve => setTimeout(resolve, 50))
    expect(mocked.getWebAccessStatus).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: 'Start' })).toBeNull()
  })

  it('treats a gateway address it cannot parse as not serving this config', async () => {
    // The gateway reports its own URL; a malformed one must not take the pane down.
    renderPane({ ...OFF, enabled: true, gateway: { reason: null, state: 'serving', url: 'http://[::1' }, mode: 'lan' })

    expect(await screen.findByText('Restart needed')).toBeTruthy()
  })

  it('waits after Stop until the restarted gateway no longer serves the web app', async () => {
    // Stop saves `enabled: false` and starts a restart. Until the old gateway exits it still serves
    // the web app; while it restarts nothing answers. The pane waits through both for the new one.
    mocked.stopWebAccess.mockResolvedValue({ ...OLD_GATEWAY, restart_started: true })
    renderPane(SERVING)
    const stop = await screen.findByRole('button', { name: 'Stop' })
    mocked.getWebAccessStatus
      .mockResolvedValueOnce(OLD_GATEWAY)
      .mockRejectedValueOnce(new Error('gateway restarting'))
      .mockResolvedValue(OFF)

    fireEvent.click(stop)

    expect(await screen.findByText('Stopped', {}, { timeout: 8000 })).toBeTruthy()
    expect(screen.queryByText('Restart needed')).toBeNull()
    expect($notifications.get()).toEqual([])
  }, 15_000)

  it('shows the web app stopping, not waiting for a restart, while the restart runs', async () => {
    mocked.stopWebAccess.mockResolvedValue({ ...OLD_GATEWAY, restart_started: true })
    renderPane(SERVING)
    const stop = await screen.findByRole('button', { name: 'Stop' })
    // The restarting gateway has not answered yet.
    mocked.getWebAccessStatus.mockReturnValue(new Promise<WebAccessStatus>(() => {}))

    fireEvent.click(stop)

    expect(await screen.findByText('Stopping')).toBeTruthy()
    expect(screen.queryByText('Restart needed')).toBeNull()
  })

  it('waits after Restart until a turned-off web app is no longer served', async () => {
    // Saved `enabled: false`, but the gateway still serves the web app: Restart applies the change.
    mocked.restartWebAccess.mockResolvedValue({ ...OLD_GATEWAY, restart_started: true })
    renderPane(OLD_GATEWAY)
    const restart = await screen.findByRole('button', { name: 'Restart gateway' })
    mocked.getWebAccessStatus.mockResolvedValueOnce(OLD_GATEWAY).mockResolvedValue(OFF)

    fireEvent.click(restart)

    expect(await screen.findByText('Stopped', {}, { timeout: 8000 })).toBeTruthy()
    expect(screen.queryByText('Restart needed')).toBeNull()
  }, 15_000)

  it('says when the gateway still serves the web app 90 seconds after Stop', async () => {
    mocked.stopWebAccess.mockResolvedValue({ ...OLD_GATEWAY, restart_started: true })
    renderPane(SERVING)
    const stop = await screen.findByRole('button', { name: 'Stop' })
    mocked.getWebAccessStatus.mockResolvedValue(OLD_GATEWAY)
    vi.useFakeTimers()

    try {
      fireEvent.click(stop)
      await vi.advanceTimersByTimeAsync(92_000)
    } finally {
      vi.useRealTimers()
    }

    await waitFor(() => expect($notifications.get()).toHaveLength(1))
    const [toast] = $notifications.get()
    expect(toast.kind).toBe('error')
    expect(toast.message).toBe("The gateway didn't stop serving the web app within 90 seconds.")
  })

  it('says that network access is not encrypted', async () => {
    // LAN mode is plain HTTP: anyone on the network can read the traffic.
    renderPane(OFF)

    expect(await screen.findByText(/The connection is not encrypted/)).toBeTruthy()
  })
})
