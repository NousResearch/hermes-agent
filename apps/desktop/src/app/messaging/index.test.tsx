// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $changeEventsAvailable, $pairingChangeTick, $platformsChangeTick } from '@/store/live-sync'
import type * as SettingsScopeModule from '@/store/settings-scope'
import { $settingsScopeOverride } from '@/store/settings-scope'
import type { MessagingPlatformInfo } from '@/types/hermes'

import { MessagingView } from './index'

type TestScope = { connectionId: string; profile?: string }

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

// Imports are static on purpose: `await import(...)` inside test bodies ran
// against the test timer, and a cold evaluate of the MessagingView graph blew
// the 15s timeout — the timed-out first test then left the DOM empty for
// every later test. vi.mock calls below are hoisted above these imports, so
// the mocks still apply.

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
const notify = vi.fn()

vi.mock('@/hermes', () => ({
  approvePairing: (platformId: string, requestId: string, profile?: unknown) =>
    approvePairing(platformId, requestId, profile),
  getMessagingPlatforms: (profile?: unknown) => getMessagingPlatforms(profile),
  getPairing: (profile?: unknown) => getPairing(profile),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  profileScopeKey: (profile?: TestScope) =>
    profile && typeof profile === 'object'
      ? `${profile.connectionId || ''}::${profile.profile || 'default'}`
      : String(profile || 'default'),
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
  notify: (notification: unknown) => notify(notification),
  notifyError: vi.fn()
}))

vi.mock('@/store/system-actions', async () => {
  const { atom } = await import('nanostores')

  return {
    $gatewayRestarting: atom(false),
    runGatewayRestart: (owner?: unknown, isCurrent?: () => boolean) => runGatewayRestart(owner, isCurrent),
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
  $settingsScopeOverride.set(null)
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

// Static import at module scope (after the hoisted vi.mock calls) so the
// heavy component-tree transform is paid during collection, not billed against
// the first test's testTimeout — inside a test body it exceeded the budget on
// loaded CI runners and cascaded the whole file (main runs 34599517793,
// 34600757569, 34601269252). Same pattern as chat/index.test.tsx.
void import('./index')

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
    $settingsScopeOverride.set(null)
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })

    await renderMessaging()

    // #118432: the backend resolves an omitted profile against the home it was
    // LAUNCHED under, so "follow the active profile" has to be said out loud
    // rather than left to the ambient fallback.
    await waitFor(() => expect(getMessagingPlatforms).toHaveBeenCalledWith(scope()))
    expect(getPairing).toHaveBeenCalledWith(scope())
  })

  it('loads custom-home messaging through its profile-less owner', async () => {
    const customOwner = { connectionId: 'local' }
    $settingsScopeOverride.set('custom')
    await setSettingsOwner(customOwner)
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform({ name: 'Custom Home Teams' })] })

    await renderMessaging()

    expect((await screen.findAllByText('Custom Home Teams')).length).toBeGreaterThan(0)
    expect(getMessagingPlatforms).toHaveBeenCalledWith(customOwner)
    expect(getPairing).toHaveBeenCalledWith(customOwner)
  })

  it('clears owner-bound state before a same-profile replacement settles, then writes through the replacement', async () => {
    const gatewayBPlatforms = deferred<{ platforms: MessagingPlatformInfo[] }>()
    const gatewayBPairing = deferred<{ approved: never[]; pending: never[] }>()
    const oldToken = 'synthetic-gateway-a-token'
    const approvedUser = { platform: 'teams', user_id: 'approved-a', user_name: 'Approved A' }

    const pendingUser = {
      age_minutes: 1,
      platform: 'teams',
      request_id: 'request-a',
      user_id: 'pending-a',
      user_name: 'Pending A'
    }

    getMessagingPlatforms.mockImplementation((owner: TestScope) =>
      owner.connectionId === 'gateway-a'
        ? Promise.resolve({
            platforms: [
              platform({
                env_vars: [
                  {
                    advanced: false,
                    description: 'Synthetic token',
                    is_password: true,
                    is_set: true,
                    key: 'TEAMS_TOKEN',
                    prompt: 'Token',
                    redacted_value: oldToken,
                    required: true,
                    url: null
                  }
                ],
                name: 'Gateway A Teams'
              })
            ]
          })
        : gatewayBPlatforms.promise
    )
    getPairing.mockImplementation((owner: TestScope) =>
      owner.connectionId === 'gateway-a'
        ? Promise.resolve({ approved: [approvedUser], pending: [pendingUser] })
        : gatewayBPairing.promise
    )

    await renderMessaging()
    expect((await screen.findAllByText('Gateway A Teams')).length).toBeGreaterThan(0)
    fireEvent.change(screen.getByLabelText('Token'), { target: { value: 'unsaved-a' } })
    fireEvent.click(await screen.findByRole('button', { name: /Revoke Approved A/ }))
    expect(screen.getByRole('dialog')).toBeTruthy()

    await act(async () => {
      await setSettingsOwner(scope('gateway-b'))
    })

    expect(screen.queryByText('Gateway A Teams')).toBeNull()
    expect(screen.queryByPlaceholderText(oldToken)).toBeNull()
    expect(screen.queryByDisplayValue('unsaved-a')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Approve' })).toBeNull()
    expect(screen.queryByRole('button', { name: /Revoke Approved A/ })).toBeNull()
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.queryByRole('switch')).toBeNull()

    await act(async () => {
      gatewayBPlatforms.resolve({ platforms: [platform({ name: 'Gateway B Teams' })] })
      gatewayBPairing.resolve({ approved: [], pending: [] })
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

describe('MessagingView status filter', () => {
  const rowNames = (container: HTMLElement) =>
    [...container.querySelectorAll('ul > li > button')].map(row => row.querySelector('.truncate')?.textContent)

  it('offers a tab only for tones some platform is in, and narrows the list to that tone', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({ enabled: true, id: 'discord', name: 'Discord', state: 'connected' }),
        platform({ enabled: true, id: 'slack', name: 'Slack', state: 'retrying' }),
        platform({ id: 'teams', name: 'Microsoft Teams' })
      ]
    })

    const { container } = await renderMessaging()

    await waitFor(() => expect(rowNames(container)).toHaveLength(3))
    expect(screen.queryByRole('button', { name: 'Errors' })).toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Needs attention' })[0])

    expect(rowNames(container)).toEqual(['Slack'])
  })
})

describe('MessagingView enable switch', () => {
  it('labels the enable switch with the platform state', async () => {
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform({ enabled: true })] })

    await renderMessaging()

    const toggle = await screen.findByRole('switch', { name: 'Disable Microsoft Teams' })
    expect(toggle.closest('label')?.textContent).toBe('Enabled')
  })

  it("drops the previous profile's credential placeholders on the very first post-switch render", async () => {
    // #96542: blanking the stale platforms state in a passive effect lets the
    // "new scope + old data" frame paint first, so the new profile briefly
    // showed the PREVIOUS profile's redacted Telegram token as the field
    // placeholder. The reset must happen during render — after the scope
    // switch returns from act(), the old value is already gone from the DOM
    // with no waitFor() in between.
    const { $settingsScopeOverride } = await import('@/store/settings-scope')
    const oldToken = '123456:AAE-previous-profile-token'

    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          env_vars: [
            {
              advanced: false,
              description: 'Telegram bot token from @BotFather.',
              is_password: true,
              is_set: true,
              key: 'TELEGRAM_TOKEN',
              prompt: 'Token',
              redacted_value: oldToken,
              required: true,
              url: null
            }
          ],
          id: 'telegram',
          name: 'Telegram'
        })
      ]
    })

    $settingsScopeOverride.set(null)
    await renderMessaging()

    // The previous profile's redacted token is visible as the placeholder.
    expect(await screen.findByPlaceholderText(oldToken)).not.toBeNull()

    // Switch the scope while B's fetch stays pending, so any stale rendering
    // would still be showing A's data.
    //
    // Note on coverage: jsdom cannot observe the paint-order race itself
    // (act() flushes passive effects synchronously, so an effect-based reset
    // also clears before the assertion; outside act() the commit itself is
    // deferred). This test therefore pins the reset *semantics* — after a
    // scope switch the previous profile's credential placeholder must be gone
    // from the DOM even while the new profile's fetch is still pending. The
    // paint-order guarantee ("the stale frame never reaches the screen") is
    // carried by resetting during render per the React docs pattern instead
    // of in a passive effect, which fires after paint.
    getMessagingPlatforms.mockReturnValue(new Promise(() => {}))

    await act(async () => {
      $settingsScopeOverride.set('profile-b')
    })

    // Synchronous assertion — no waitFor: the reset must have discarded the
    // stale platforms state before the new profile's fetch resolves.
    expect(screen.queryByPlaceholderText(oldToken)).toBeNull()

    // Let pending work settle and restore the shared store.
    await act(async () => {
      $settingsScopeOverride.set(null)
    })
  })

  it("drops profile A's late response after the scope switched to B", async () => {
    // #96542 (second mechanism): A's in-flight getMessagingPlatforms resolves
    // AFTER the switch to B and must not repaint A's redacted token under B.
    const tokenA = '123456:AAE-profile-a-token'

    let resolveA: (value: unknown) => void = () => {}

    getMessagingPlatforms.mockImplementation((profile?: null | string) =>
      profile === 'profile-b'
        ? new Promise(() => {})
        : new Promise(resolve => {
            resolveA = resolve
          })
    )

    $settingsScopeOverride.set(null)
    await renderMessaging()

    await act(async () => {
      $settingsScopeOverride.set('profile-b')
    })

    await act(async () => {
      resolveA({
        platforms: [
          platform({
            env_vars: [
              {
                advanced: false,
                description: 'Telegram bot token from @BotFather.',
                is_password: true,
                is_set: true,
                key: 'TELEGRAM_TOKEN',
                prompt: 'Token',
                redacted_value: tokenA,
                required: true,
                url: null
              }
            ],
            id: 'telegram',
            name: 'Telegram'
          })
        ]
      })
    })

    expect(screen.queryByPlaceholderText(tokenA)).toBeNull()

    getMessagingPlatforms.mockReset()
    await act(async () => {
      $settingsScopeOverride.set(null)
    })
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
    expect(runGatewayRestart).toHaveBeenNthCalledWith(1, scope(), expect.any(Function))
    expect(runGatewayRestart).toHaveBeenNthCalledWith(2, scope(), expect.any(Function))
  })
})

describe('MessagingView allowlist editor', () => {
  const allowlist = (patch: Record<string, unknown> = {}) => ({
    advanced: false,
    description: 'Allowed users',
    is_list: true,
    is_password: false,
    is_set: true,
    key: 'TEAMS_ALLOWED_USERS',
    prompt: 'Allowed users',
    redacted_value: '«redacted:111...222»',
    required: false,
    url: null,
    value: '111,222',
    ...patch
  })

  const entries = () =>
    screen.getAllByRole('textbox').filter(el => /^Allowed users \d+$/.test(el.getAttribute('aria-label') || ''))

  async function save() {
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Save changes/ }))
    })
  }

  it('shows each saved ID in its own visible box and saves add/remove edits as one list', async () => {
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform({ env_vars: [allowlist()] })] })
    await renderMessaging()

    await screen.findByLabelText('Allowed users 1')
    expect(entries().map(el => [(el as HTMLInputElement).type, (el as HTMLInputElement).value])).toEqual([
      ['text', '111'],
      ['text', '222']
    ])

    fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0])
    fireEvent.click(screen.getByRole('button', { name: /Add another/ }))
    // A pasted comma list splits into one box per entry.
    fireEvent.change(entries()[1], { target: { value: '333, 444' } })
    expect(entries().map(el => (el as HTMLInputElement).value)).toEqual(['222', '333', '444'])

    await save()
    expect(updateMessagingPlatform).toHaveBeenCalledWith(
      'teams',
      { env: { TEAMS_ALLOWED_USERS: '222,333,444' } },
      scope('gateway-a')
    )
  })

  it('clears a saved allowlist when every entry is removed', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [platform({ env_vars: [allowlist({ redacted_value: '«redacted:111»', value: '111' })] })]
    })
    await renderMessaging()

    await screen.findByLabelText('Allowed users 1')
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
    await save()
    expect(updateMessagingPlatform).toHaveBeenCalledWith(
      'teams',
      { clear_env: ['TEAMS_ALLOWED_USERS'], env: {} },
      scope('gateway-a')
    )
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
      bot_username: 'hermes_bot',
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
      expect(notify).toHaveBeenCalledWith({
        kind: 'success',
        message: 'Connected: @hermes_bot · Telegram saved; gateway restarting…',
        title: 'Telegram setup saved'
      })
      // The pairing UI is gone, but the card still names the bot that was just connected.
      expect(screen.queryByRole('button', { name: /Save and restart/ })).toBeNull()
      expect(screen.getByText('Connected')).toBeTruthy()
      expect(screen.getByText('@hermes_bot')).toBeTruthy()
    } finally {
      $settingsScopeOverride.set(null)
    }
  })
})
