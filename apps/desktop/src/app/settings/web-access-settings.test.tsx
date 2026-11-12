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

  // `hermes config set webapp.*` (or another window) changes the status while the pane is open.
  const EDITED_ELSEWHERE: WebAccessStatus = { ...OFF, enabled: true, mode: 'public', public_url: 'https://hermes.example.org' }

  it('picks up a web access change saved elsewhere when the window regains focus', async () => {
    renderPane(OFF)
    expect(await screen.findByText('Stopped')).toBeTruthy()
    mocked.getWebAccessStatus.mockResolvedValue(EDITED_ELSEWHERE)

    fireEvent.focus(window)

    expect(await screen.findByText('Restart needed')).toBeTruthy()
    // The address choice follows the saved config, not the one the pane opened with.
    expect(screen.getByDisplayValue('https://hermes.example.org')).toBeTruthy()
  })

  it('keeps an unsaved address choice through a refetch that changes nothing', async () => {
    renderPane(OFF)
    fireEvent.click(await screen.findByRole('button', { name: 'My own URL' }))
    fireEvent.change(screen.getByPlaceholderText('https://hermes.example.com'), {
      target: { value: 'https://typed.example.net' }
    })
    mocked.getWebAccessStatus.mockClear()

    fireEvent.focus(window)

    await waitFor(() => expect(mocked.getWebAccessStatus).toHaveBeenCalled())
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(screen.getByDisplayValue('https://typed.example.net')).toBeTruthy()
  })

  it('keeps reading the status while the pane stays open', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })

    try {
      renderPane(OFF)
      expect(await screen.findByText('Stopped')).toBeTruthy()
      mocked.getWebAccessStatus.mockResolvedValue(EDITED_ELSEWHERE)

      vi.advanceTimersByTime(30_000)

      expect(await screen.findByText('Restart needed')).toBeTruthy()
    } finally {
      vi.useRealTimers()
    }
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

  it('frees the controls once a restarting Start has gone unconfirmed as long as the restart wait', async () => {
    // Start answered that it is restarting the gateway. The backend answering is that gateway, and
    // no replacement ever answers, so the Start's answer is the last status the pane has.
    const pending: WebAccessStatus = { ...OFF, enabled: true, mode: 'lan' }
    mocked.startWebAccess.mockResolvedValue({
      ...pending,
      restart_pid: 9,
      restart_started: true,
      start_progress: { started_at: Date.now() / 1000, step: 'restarting' }
    })
    renderPane(OFF)
    const start = await screen.findByRole('button', { name: 'Start' })
    mocked.getWebAccessStatus.mockRejectedValue(new Error('gateway restarting'))
    vi.useFakeTimers()

    try {
      fireEvent.click(start)
      await vi.advanceTimersByTimeAsync(92_000)
    } finally {
      vi.useRealTimers()
    }

    expect((await screen.findByRole('button', { name: 'Restart gateway' })).hasAttribute('disabled')).toBe(false)
    expect(screen.queryByText('Restarting the gateway to serve the web app.')).toBeNull()
  })

  it('names the cause when the restart exits without replacing the gateway', async () => {
    // The restart answered `restart_started` with its pid, then exited at once: the old gateway keeps
    // serving the old config. An older restart's failure (another pid) is not this one's.
    const pending: WebAccessStatus = { ...OFF, enabled: true, mode: 'lan' }
    const failure = (pid: number, reason: string) => ({ exit_code: 1, pid, reason })
    mocked.startWebAccess.mockResolvedValue({ ...pending, restart_pid: 4242, restart_started: true })
    renderPane(OFF)
    const start = await screen.findByRole('button', { name: 'Start' })
    mocked.getWebAccessStatus
      .mockResolvedValueOnce({ ...pending, restart_failure: failure(1111, 'an older restart') })
      .mockResolvedValue({ ...pending, restart_failure: failure(4242, 'run `hermes pm repair`') })

    fireEvent.click(start)

    await waitFor(() => expect($notifications.get()).toHaveLength(1), { timeout: 8000 })
    const [toast] = $notifications.get()
    expect(toast.kind).toBe('error')
    expect(toast.message).toBe("The gateway couldn't restart: run `hermes pm repair`")
  }, 15_000)

  it('names the cause when a Stop restart fails while the gateway could not serve the web app', async () => {
    // Saved `enabled: false` reads as applied at once (the gateway serves nothing), but the restart
    // the user asked for exited.
    const unavailable: WebAccessStatus = { ...OFF, gateway: { reason: 'port 9120 is in use', state: 'unavailable', url: null } }
    mocked.stopWebAccess.mockResolvedValue({ ...unavailable, restart_pid: 7, restart_started: true })
    renderPane({ ...unavailable, enabled: true, mode: 'lan' })
    const stop = await screen.findByRole('button', { name: 'Stop' })
    mocked.getWebAccessStatus.mockResolvedValue({
      ...unavailable,
      restart_failure: { exit_code: 1, pid: 7, reason: 'run `hermes pm repair`' }
    })

    fireEvent.click(stop)

    await waitFor(() => expect($notifications.get()).toHaveLength(1), { timeout: 8000 })
    expect($notifications.get()[0].message).toBe("The gateway couldn't restart: run `hermes pm repair`")
  }, 15_000)

  it('shows no LAN address under My own URL while the URL is empty', async () => {
    renderPane(OFF)
    fireEvent.click(await screen.findByRole('button', { name: 'My own URL' }))

    expect(screen.queryByText(/192\.168\.1\.50/)).toBeNull()

    fireEvent.change(screen.getByPlaceholderText('https://hermes.example.com'), {
      target: { value: 'https://hermes.tail.ts.net' }
    })

    expect(screen.getByText('https://hermes.tail.ts.net')).toBeTruthy()
  })

  it('says that network access is not encrypted', async () => {
    // LAN mode is plain HTTP: anyone on the network can read the traffic.
    renderPane(OFF)

    expect(await screen.findByText(/The connection is not encrypted/)).toBeTruthy()
  })
})

describe('WebAccessSettings while Start runs', () => {
  // A backend that reports a Start in flight answers `start_progress` (null when none runs).
  const IDLE: WebAccessStatus = { ...OFF, start_progress: null }

  const building = (secondsAgo = 0): WebAccessStatus => ({
    ...IDLE,
    start_progress: { started_at: Date.now() / 1000 - secondsAgo, step: 'building' }
  })

  // The progress poll and the elapsed clock run only while the window is viewed. `Date` is pinned:
  // the statuses and the elapsed clock both read it, so a runner pause cannot move the label.
  beforeEach(() => {
    vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(true)
    vi.useFakeTimers({ now: new Date('2026-11-03T12:00:00Z'), toFake: ['Date', 'setInterval', 'clearInterval'] })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.mocked(globalThis.document.hasFocus).mockRestore()
  })

  it('says which step Start is on and how long it has run while it builds', async () => {
    // The build can take minutes: Start's request answers only when it is done.
    mocked.startWebAccess.mockReturnValue(new Promise(() => {}))
    renderPane(IDLE)
    const start = await screen.findByRole('button', { name: 'Start' })
    mocked.getWebAccessStatus.mockResolvedValue(building())

    fireEvent.click(start)

    expect(await screen.findByText('Building the web app. This can take a few minutes.')).toBeTruthy()
    expect(screen.getByText('Starting')).toBeTruthy()
    expect(screen.getByText(/\d+s elapsed/)).toBeTruthy()

    mocked.getWebAccessStatus.mockResolvedValue({
      ...IDLE,
      start_progress: { started_at: Date.now() / 1000, step: 'registering' }
    })
    vi.advanceTimersByTime(1000)

    expect(await screen.findByText('Setting up Nous sign-in for this address.')).toBeTruthy()
  })

  it('shows a Start already running when the pane opens during it, and its result when it ends', async () => {
    // Another window started web access, or this pane was closed and reopened while it built.
    renderPane(building(125))

    expect(await screen.findByText('Building the web app. This can take a few minutes.')).toBeTruthy()
    expect(screen.getByText('2:05 elapsed')).toBeTruthy()
    expect(screen.getByText('Starting')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Start' }).hasAttribute('disabled')).toBe(true)

    mocked.getWebAccessStatus.mockResolvedValue({ ...SERVING, start_progress: null })
    vi.advanceTimersByTime(1000)

    expect(await screen.findByRole('button', { name: 'Stop' })).toBeTruthy()
    expect(screen.queryByText(/elapsed/)).toBeNull()
    expect(mocked.startWebAccess).not.toHaveBeenCalled()
  })

  it('says the gateway is restarting once Start has saved the change', async () => {
    mocked.startWebAccess.mockResolvedValue({
      ...OFF,
      enabled: true,
      mode: 'lan',
      restart_pid: 9,
      restart_started: true
    })
    renderPane(IDLE)
    const start = await screen.findByRole('button', { name: 'Start' })
    // The restarting gateway has not answered yet.
    mocked.getWebAccessStatus.mockReturnValue(new Promise<WebAccessStatus>(() => {}))

    fireEvent.click(start)

    expect(await screen.findByText('Restarting the gateway to serve the web app.')).toBeTruthy()
  })

  it('shows a Start restarting in a pane opened after its request answered, until the gateway serves it', async () => {
    // Start saved the config and asked for a restart; the old gateway still answers, serving no
    // web app. That is a Start under way, not a saved change waiting for someone to restart it.
    renderPane({
      ...SERVING,
      gateway: OFF.gateway,
      start_progress: { started_at: Date.now() / 1000 - 40, step: 'restarting' }
    })

    expect(await screen.findByText('Restarting the gateway to serve the web app.')).toBeTruthy()
    expect(screen.getByText('40s elapsed')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Restart gateway' }).hasAttribute('disabled')).toBe(true)

    mocked.getWebAccessStatus.mockResolvedValue({ ...SERVING, start_progress: null })
    vi.advanceTimersByTime(1000)

    expect(await screen.findByRole('button', { name: 'Stop' })).toBeTruthy()
  })

  it('says a Start is getting ready while another Web Access change goes first', async () => {
    renderPane({ ...IDLE, start_progress: { started_at: Date.now() / 1000, step: 'preparing' } })

    expect(await screen.findByText('Getting the web app ready. Building it can take a few minutes.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Start' }).hasAttribute('disabled')).toBe(true)
  })

  it('describes the Start of a backend that reports no step without polling it for one', async () => {
    // A backend older than `start_progress`: its status never carries the field.
    mocked.startWebAccess.mockReturnValue(new Promise(() => {}))
    renderPane(OFF)
    fireEvent.click(await screen.findByRole('button', { name: 'Start' }))

    expect(await screen.findByText('Getting the web app ready. Building it can take a few minutes.')).toBeTruthy()
    const calls = mocked.getWebAccessStatus.mock.calls.length
    vi.advanceTimersByTime(3000)
    await new Promise(resolve => setTimeout(resolve, 20))

    expect(mocked.getWebAccessStatus.mock.calls.length).toBe(calls)
  })
})
