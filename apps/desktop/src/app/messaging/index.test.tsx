// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as SettingsScopeModule from '@/store/settings-scope'
import type { MessagingPlatformInfo } from '@/types/hermes'

type TestScope = { connectionId: string; profile: string }

const scope = (connectionId = 'gateway-a', profile = 'default'): TestScope => ({ connectionId, profile })

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

async function setSettingsOwner(value: TestScope) {
  const { $settingsOwner } = await import('@/store/settings-scope')
  const owner = $settingsOwner as unknown as { set: (scope: TestScope) => void }

  owner.set(value)
}

const getMessagingPlatforms = vi.fn()
const updateMessagingPlatform = vi.fn()
const getPairing = vi.fn()
const approvePairing = vi.fn()
const revokePairing = vi.fn()
const openExternalLink = vi.fn()
const runGatewayRestart = vi.fn()
const watchGatewayRestartOutcome = vi.fn()
const startTelegramOnboarding = vi.fn()
const getTelegramOnboardingStatus = vi.fn()
const applyTelegramOnboarding = vi.fn()

vi.mock('@/hermes', () => ({
  approvePairing: (platformId: string, requestId: string, profile?: unknown) =>
    approvePairing(platformId, requestId, profile),
  getMessagingPlatforms: (profile?: unknown) => getMessagingPlatforms(profile),
  getPairing: (profile?: unknown) => getPairing(profile),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  revokePairing: (platformId: string, userId: string, profile?: unknown) =>
    revokePairing(platformId, userId, profile),
  setApiRequestProfile: vi.fn(),
  applyTelegramOnboarding: (pairingId: string, ids: string[], profile?: unknown) =>
    applyTelegramOnboarding(pairingId, ids, profile),
  cancelTelegramOnboarding: vi.fn(async () => ({ ok: true })),
  getTelegramOnboardingStatus: (pairingId: string, profile?: unknown) =>
    getTelegramOnboardingStatus(pairingId, profile),
  startTelegramOnboarding: (botName?: string, profile?: unknown) => startTelegramOnboarding(botName, profile),
  updateMessagingPlatform: (id: string, body: unknown, profile?: unknown) =>
    updateMessagingPlatform(id, body, profile)
}))

vi.mock('@/store/settings-scope', async importOriginal => {
  const actual = await importOriginal<typeof SettingsScopeModule>()
  const { atom } = await import('nanostores')

  return { ...actual, $settingsOwner: atom({ connectionId: 'gateway-a', profile: 'default' }) }
})

vi.mock('qrcode', () => ({ toDataURL: vi.fn(async () => 'data:image/png;base64,QR') }))

// Keep store/profile's side-effecting imports inert (pulled in via the shared
// settings scope store) — same seam as store/profile.test.ts.
vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => null, subscribe: () => () => {} },
  ensureGatewayForAgent: vi.fn(async () => undefined),
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/system-actions', async () => {
  const { atom } = await import('nanostores')

  return {
    $gatewayRestarting: atom(false),
    runGatewayRestart: (owner?: unknown) => runGatewayRestart(owner),
    watchGatewayRestartOutcome: (owner?: unknown) => watchGatewayRestartOutcome(owner)
  }
})

function platform(patch: Partial<MessagingPlatformInfo> = {}): MessagingPlatformInfo {
  return {
    configured: false,
    description: 'A platform.',
    docs_url: '',
    enabled: false,
    env_vars: [],
    gateway_running: true,
    id: 'teams',
    name: 'Microsoft Teams',
    state: 'disabled',
    ...patch
  }
}

beforeEach(async () => {
  await setSettingsOwner(scope())
  updateMessagingPlatform.mockResolvedValue({ ok: true, platform: 'teams' })
  getPairing.mockResolvedValue({ approved: [], pending: [] })
  runGatewayRestart.mockResolvedValue(true)
  watchGatewayRestartOutcome.mockResolvedValue(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// Import at module scope (after the hoisted vi.mock calls) so the heavy
// component-tree transform is paid during collection, not billed against the
// first test's testTimeout — inside a test body it exceeded the budget on
// loaded CI runners and cascaded the whole file (main runs 34599517793,
// 34600757569, 34601269252). Same pattern as chat/index.test.tsx.
const { MessagingView } = await import('./index')

async function renderMessaging() {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <MemoryRouter>
        <MessagingView />
      </MemoryRouter>
    )
  })

  return result!
}

describe('MessagingView profile scope', () => {
  it('names the active profile explicitly instead of sending an unscoped request', async () => {
    const { $settingsScopeOverride } = await import('@/store/settings-scope')

    $settingsScopeOverride.set(null)
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })

    await renderMessaging()

    // #118432: the backend resolves an omitted profile against the home it was
    // LAUNCHED under, so "follow the active profile" has to be said out loud
    // rather than left to the ambient fallback.
    await waitFor(() => expect(getMessagingPlatforms).toHaveBeenCalledWith(scope()))
    expect(getPairing).toHaveBeenCalledWith(scope())
  })

  it('refetches and writes through the replacement gateway owner when the profile name is unchanged', async () => {
    getMessagingPlatforms.mockImplementation((owner: TestScope) =>
      Promise.resolve({
        platforms: [platform({ name: owner.connectionId === 'gateway-a' ? 'Gateway A Teams' : 'Gateway B Teams' })]
      })
    )

    await renderMessaging()
    expect((await screen.findAllByText('Gateway A Teams')).length).toBeGreaterThan(0)

    await act(async () => {
      await setSettingsOwner(scope('gateway-b'))
    })

    expect((await screen.findAllByText('Gateway B Teams')).length).toBeGreaterThan(0)
    await act(async () => {
      fireEvent.click(screen.getByRole('switch'))
    })

    await waitFor(() =>
      expect(updateMessagingPlatform).toHaveBeenCalledWith('teams', { enabled: true }, scope('gateway-b'))
    )
  })

  it('ignores a late read from the replaced gateway owner', async () => {
    const gatewayA = deferred<{ platforms: MessagingPlatformInfo[] }>()
    getMessagingPlatforms.mockImplementation((owner: TestScope) =>
      owner.connectionId === 'gateway-a'
        ? gatewayA.promise
        : Promise.resolve({ platforms: [platform({ name: 'Gateway B Teams' })] })
    )

    await renderMessaging()
    await waitFor(() => expect(getMessagingPlatforms).toHaveBeenCalledWith(scope('gateway-a')))
    await act(async () => setSettingsOwner(scope('gateway-b')))
    expect((await screen.findAllByText('Gateway B Teams')).length).toBeGreaterThan(0)

    await act(async () => gatewayA.resolve({ platforms: [platform({ name: 'Late Gateway A Teams' })] }))
    expect(screen.queryByText('Late Gateway A Teams')).toBeNull()
  })

  it('does not apply a late mutation completion to the replacement owner', async () => {
    const update = deferred<{ hot_served: boolean }>()
    getMessagingPlatforms.mockImplementation((owner: TestScope) =>
      Promise.resolve({
        platforms: [platform({ name: owner.connectionId === 'gateway-a' ? 'Gateway A Teams' : 'Gateway B Teams' })]
      })
    )
    updateMessagingPlatform.mockReturnValue(update.promise)

    await renderMessaging()
    await screen.findAllByText('Gateway A Teams')
    fireEvent.click(screen.getByRole('switch'))
    await waitFor(() => expect(updateMessagingPlatform).toHaveBeenCalledWith('teams', { enabled: true }, scope('gateway-a')))
    await act(async () => setSettingsOwner(scope('gateway-b')))
    expect((await screen.findAllByText('Gateway B Teams')).length).toBeGreaterThan(0)

    await act(async () => update.resolve({ hot_served: false }))
    expect(screen.queryByText('Gateway A Teams')).toBeNull()
    expect(runGatewayRestart).not.toHaveBeenCalled()
  })
})

describe('MessagingView setup-guide link', () => {
  it('hides the setup-guide button for a plugin platform with no docs URL', async () => {
    // Teams (and other plugin platforms) ship an empty docs_url. Rendering an
    // anchor with href="" let Electron resolve it to the app's own packaged
    // index.html and fail with an OS "file not found" dialog. The button must
    // simply not appear when there is no guide to open.
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform({ docs_url: '' })] })

    await renderMessaging()

    expect((await screen.findAllByText('Microsoft Teams')).length).toBeGreaterThan(0)
    expect(screen.queryByText('Open setup guide')).toBeNull()
  })

  it('opens a real docs URL through the validated external opener', async () => {
    const docsUrl = 'https://hermes-agent.nousresearch.com/docs/user-guide/messaging/teams'
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform({ docs_url: docsUrl })] })

    await renderMessaging()

    const link = await screen.findByText('Open setup guide')
    await act(async () => {
      fireEvent.click(link)
    })

    await waitFor(() => expect(openExternalLink).toHaveBeenCalledWith(docsUrl))
  })
})

describe('MessagingView pairing', () => {
  const pendingUser = {
    age_minutes: 3,
    platform: 'teams',
    request_id: 'a1b2c3d4e5f60718',
    user_id: '7712345',
    user_name: 'Bee'
  }

  it('approves the listed request by its request id, never by a code', async () => {
    // The whole point of the request-id grant path: the UI can only ever send
    // the server-side row id, because the one-time code is never returned by
    // the API. Posting anything derived from the code could not be approved.
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })
    getPairing.mockResolvedValue({ approved: [], pending: [pendingUser] })
    approvePairing.mockResolvedValue({ ok: true, user: { user_id: '7712345', user_name: 'Bee' } })

    await renderMessaging()

    const approve = await screen.findByRole('button', { name: 'Approve' })
    await act(async () => {
      fireEvent.click(approve)
    })

    await waitFor(() => expect(approvePairing).toHaveBeenCalledWith('teams', 'a1b2c3d4e5f60718', scope()))
  })

  it('restores the pending row when approval fails', async () => {
    // Optimistic removal must not silently swallow the request: a failed
    // approve has to leave the operator something to retry.
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })
    getPairing.mockResolvedValue({ approved: [], pending: [pendingUser] })
    approvePairing.mockRejectedValue(new Error('500 boom'))

    await renderMessaging()

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: 'Approve' }))
    })

    expect(await screen.findByRole('button', { name: 'Approve' })).toBeTruthy()
    expect(screen.getByText('Bee')).toBeTruthy()
  })

  it('shows no pairing affordance when nobody is waiting', async () => {
    // Approvals are rare; an always-present empty state would be permanent
    // chrome on a page that is otherwise about credentials.
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })
    getPairing.mockResolvedValue({ approved: [], pending: [] })

    await renderMessaging()

    expect((await screen.findAllByText('Microsoft Teams')).length).toBeGreaterThan(0)
    expect(screen.queryByRole('button', { name: 'Approve' })).toBeNull()
    expect(screen.queryByText(/Pending requests/)).toBeNull()
  })

  it('still renders platforms when the pairing endpoint fails', async () => {
    // An older backend without the endpoint must not blank the page.
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })
    getPairing.mockRejectedValue(new Error('404 not found'))

    await renderMessaging()

    expect((await screen.findAllByText('Microsoft Teams')).length).toBeGreaterThan(0)
    expect(screen.queryByRole('button', { name: 'Approve' })).toBeNull()
  })

  it('refetches pending rows on pairing.changed, not on platforms.changed', async () => {
    // The two signals are not interchangeable: platforms.changed tracks
    // connect/disconnect health via gateway_state.json, which a new pairing
    // request never moves. Riding it would leave someone invisible in the
    // pending list until an unrelated reconnect happened to fire.
    const { $changeEventsAvailable, $pairingChangeTick, $platformsChangeTick } = await import('@/store/live-sync')

    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })
    getPairing.mockResolvedValue({ approved: [], pending: [] })

    await renderMessaging()
    await act(async () => {
      $changeEventsAvailable.set(true)
    })
    getPairing.mockClear()

    // Someone DMs the bot: the store moves, the watcher ticks pairing.changed.
    getPairing.mockResolvedValue({ approved: [], pending: [pendingUser] })
    await act(async () => {
      $pairingChangeTick.set($pairingChangeTick.get() + 1)
    })

    await waitFor(() => expect(getPairing).toHaveBeenCalled())
    expect(await screen.findByRole('button', { name: 'Approve' })).toBeTruthy()

    // A platform health tick alone must not be what fetches pairing.
    getPairing.mockClear()
    await act(async () => {
      $platformsChangeTick.set($platformsChangeTick.get() + 1)
    })
    expect(getPairing).not.toHaveBeenCalled()
  })
})

describe('MessagingView restart banner', () => {
  const tokenField = {
    advanced: false,
    description: 'Bot token',
    is_password: true,
    is_set: false,
    key: 'TEAMS_TOKEN',
    prompt: 'Token',
    redacted_value: null,
    required: true,
    url: null
  }

  it('keeps a restart banner up after a save until the restart completes', async () => {
    // A toast vanishes; the credential still only loads on the next gateway
    // start. The page must keep saying so — and only stop once a restart
    // actually succeeded, not merely because one was requested.
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform({ env_vars: [tokenField] })] })
    runGatewayRestart.mockResolvedValueOnce(false).mockResolvedValueOnce(true)

    await renderMessaging()

    fireEvent.change(await screen.findByLabelText('Token'), { target: { value: 'secret-1' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Save changes/ }))
    })

    await waitFor(() => expect(updateMessagingPlatform).toHaveBeenCalled())
    const restartNow = await screen.findByRole('button', { name: 'Restart now' })

    await act(async () => {
      fireEvent.click(restartNow)
    })
    // First attempt failed: banner stays.
    expect(await screen.findByRole('button', { name: 'Restart now' })).toBeTruthy()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Restart now' }))
    })
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Restart now' })).toBeNull())
    expect(runGatewayRestart).toHaveBeenCalledTimes(2)
    expect(runGatewayRestart).toHaveBeenNthCalledWith(1, scope())
    expect(runGatewayRestart).toHaveBeenNthCalledWith(2, scope())
  })
})

describe('MessagingView Telegram quick setup', () => {
  it('ignores a late setup result from the replaced gateway owner', async () => {
    const start = deferred<{
      deep_link: string
      expires_at: string
      pairing_id: string
      qr_payload: string
      suggested_username: string
    }>()

    getMessagingPlatforms.mockResolvedValue({
      platforms: [platform({ id: 'telegram', name: 'Telegram', state: 'not_configured' })]
    })
    startTelegramOnboarding.mockReturnValue(start.promise)

    await renderMessaging()
    fireEvent.click(await screen.findByRole('button', { name: /Create with QR/ }))
    await waitFor(() => expect(startTelegramOnboarding).toHaveBeenCalledWith(undefined, scope('gateway-a')))
    await act(async () => setSettingsOwner(scope('gateway-b')))
    await act(async () =>
      start.resolve({
        deep_link: 'https://t.me/BotFather?start=stale',
        expires_at: new Date(Date.now() + 600_000).toISOString(),
        pairing_id: 'stale-pair',
        qr_payload: 'tg://stale',
        suggested_username: 'stale_bot'
      })
    )

    expect(getTelegramOnboardingStatus).not.toHaveBeenCalledWith('stale-pair', scope('gateway-a'))
    expect(screen.getByRole('button', { name: /Create with QR/ })).toBeTruthy()
  })

  it('runs the QR pairing to apply on the page scope and watches the backend restart', async () => {
    // Every call of one pairing must hit the SAME backend (the pairing lives in
    // that process's memory), so start/status/apply all carry the page's scope
    // and apply names the profile the credentials land in.
    const { $settingsScopeOverride } = await import('@/store/settings-scope')
    $settingsScopeOverride.set('worker')
    await setSettingsOwner(scope('gateway-a', 'worker'))
    getMessagingPlatforms.mockResolvedValue({
      platforms: [platform({ id: 'telegram', name: 'Telegram', state: 'not_configured' })]
    })
    startTelegramOnboarding.mockResolvedValue({
      deep_link: 'https://t.me/BotFather?start=abc',
      expires_at: new Date(Date.now() + 600_000).toISOString(),
      pairing_id: 'pair-1',
      qr_payload: 'tg://pair',
      suggested_username: 'hermes_bot'
    })
    getTelegramOnboardingStatus.mockResolvedValue({
      bot_username: 'hermes_bot',
      expires_at: new Date(Date.now() + 600_000).toISOString(),
      owner_user_id: '8792111505',
      status: 'ready'
    })
    applyTelegramOnboarding.mockResolvedValue({
      needs_restart: false,
      ok: true,
      platform: 'telegram',
      restart_started: true
    })

    try {
      await renderMessaging()

      await act(async () => {
        fireEvent.click(await screen.findByRole('button', { name: /Create with QR/ }))
      })
      await waitFor(() => expect(startTelegramOnboarding).toHaveBeenCalledWith(undefined, scope('gateway-a', 'worker')))

      const save = await screen.findByRole('button', { name: /Save and restart/ }, { timeout: 4000 })
      expect(getTelegramOnboardingStatus).toHaveBeenCalledWith('pair-1', scope('gateway-a', 'worker'))
      expect(screen.getByText('8792111505')).toBeTruthy()

      await act(async () => {
        fireEvent.click(save)
      })

      await waitFor(() =>
        expect(applyTelegramOnboarding).toHaveBeenCalledWith('pair-1', ['8792111505'], scope('gateway-a', 'worker'))
      )
      await waitFor(() => expect(watchGatewayRestartOutcome).toHaveBeenCalledWith(scope('gateway-a', 'worker')))
    } finally {
      $settingsScopeOverride.set(null)
    }
  })
})
