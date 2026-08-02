// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider, type Locale } from '@/i18n'
import type { MessagingPlatformInfo } from '@/types/hermes'

const getMessagingPlatforms = vi.fn()
const getEmailAutoReplyPolicy = vi.fn()
const updateMessagingPlatform = vi.fn()
const updateEmailAutoReplyPolicy = vi.fn()
const getPairing = vi.fn()
const approvePairing = vi.fn()
const revokePairing = vi.fn()
const openExternalLink = vi.fn()
const notifyError = vi.fn()

vi.mock('@/hermes', () => ({
  approvePairing: (platformId: string, requestId: string) => approvePairing(platformId, requestId),
  getEmailAutoReplyPolicy: () => getEmailAutoReplyPolicy(),
  getMessagingPlatforms: () => getMessagingPlatforms(),
  getPairing: () => getPairing(),
  revokePairing: (platformId: string, userId: string) => revokePairing(platformId, userId),
  updateEmailAutoReplyPolicy: (body: unknown) => updateEmailAutoReplyPolicy(body),
  updateMessagingPlatform: (id: string, body: unknown) => updateMessagingPlatform(id, body)
}))

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: (error: unknown, title: string) => notifyError(error, title)
}))

vi.mock('@/store/system-actions', () => ({
  runGatewayRestart: vi.fn()
}))

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

beforeEach(() => {
  getEmailAutoReplyPolicy.mockResolvedValue({
    fields: [
      {
        advanced: false,
        current_value: 'false',
        default_value: 'false',
        description: '',
        input_type: 'boolean',
        is_password: false,
        is_set: false,
        key: 'auto_reply_promotions',
        prompt: '',
        redacted_value: null,
        required: false,
        url: null
      },
      {
        advanced: false,
        current_value: '',
        default_value: '',
        description: '',
        input_type: 'textarea',
        is_password: false,
        is_set: false,
        key: 'no_reply_keywords',
        prompt: '',
        redacted_value: null,
        required: false,
        url: null
      },
      {
        advanced: false,
        current_value: '',
        default_value: '',
        description: '',
        input_type: 'textarea',
        is_password: false,
        is_set: false,
        key: 'skip_patterns',
        prompt: '',
        redacted_value: null,
        required: false,
        url: null
      },
      {
        advanced: false,
        current_value: '',
        default_value: '',
        description: '',
        input_type: 'textarea',
        is_password: false,
        is_set: false,
        key: 'force_reply_keywords',
        prompt: '',
        redacted_value: null,
        required: false,
        url: null
      },
      {
        advanced: false,
        current_value: 'true',
        default_value: 'true',
        description: '',
        input_type: 'boolean',
        is_password: false,
        is_set: false,
        key: 'require_structured_response',
        prompt: '',
        redacted_value: null,
        required: false,
        url: null
      }
    ],
    policy: {
      auto_reply_calendar: false,
      auto_reply_newsletters: false,
      auto_reply_promotions: false,
      auto_reply_reports: false,
      auto_reply_security: false,
      auto_reply_social: false,
      auto_reply_transactions: false,
      force_reply_keywords: '',
      no_reply_keywords: '',
      skip_patterns: '',
      require_structured_response: true
    }
  })
  updateEmailAutoReplyPolicy.mockResolvedValue({ ok: true })
  updateMessagingPlatform.mockResolvedValue({ ok: true, platform: 'teams' })
  getPairing.mockResolvedValue({ approved: [], pending: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderMessaging(locale: Locale = 'en') {
  const { MessagingView } = await import('./index')
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={null} initialLocale={locale}>
        <MemoryRouter>
          <MessagingView />
        </MemoryRouter>
      </I18nProvider>
    )
  })

  return result!
}

describe('MessagingView email policy', () => {
  it('renders the policy as a dedicated localized control surface', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          configured: true,
          id: 'email',
          name: 'Email',
          env_vars: [
            {
              advanced: false,
              current_value: 'bot@example.com',
              description: 'Backend English description',
              input_type: 'text',
              is_password: false,
              is_set: true,
              key: 'EMAIL_ADDRESS',
              prompt: 'Backend English mailbox prompt',
              redacted_value: null,
              required: true,
              url: null
            }
          ]
        })
      ]
    })

    await renderMessaging('zh')

    expect(await screen.findByText('自动回复策略')).toBeTruthy()
    expect(screen.getByText('推广与营销')).toBeTruthy()
    expect(screen.getByText('绝不回复')).toBeTruthy()
    expect(screen.getByText('跳过正则表达式')).toBeTruthy()
    expect(screen.getByText('要求结构化回复判定')).toBeTruthy()
    expect(screen.getByText('邮箱地址')).toBeTruthy()
    expect(screen.queryByText('Backend English promotion prompt')).toBeNull()

    fireEvent.change(screen.getByLabelText('绝不回复'), {
      target: { value: 'invoice + overdue\nvip customer' }
    })
    fireEvent.click(screen.getByRole('button', { name: '保存更改' }))

    await waitFor(() =>
      expect(updateEmailAutoReplyPolicy).toHaveBeenCalledWith({
        values: { no_reply_keywords: 'invoice + overdue;vip customer' },
        clear: []
      })
    )
  })

  it('saves skip patterns through the dedicated policy endpoint', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [platform({ configured: true, id: 'email', name: 'Email', env_vars: [] })]
    })

    await renderMessaging('zh')

    fireEvent.change(await screen.findByLabelText('跳过正则表达式'), {
      target: { value: '^out of office$' }
    })
    fireEvent.click(screen.getByRole('button', { name: '保存更改' }))

    await waitFor(() =>
      expect(updateEmailAutoReplyPolicy).toHaveBeenCalledWith({
        values: { skip_patterns: '^out of office$' },
        clear: []
      })
    )
  })

  it('rejects the same semantic keyword group on both reply lists', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          configured: true,
          id: 'email',
          name: 'Email',
          env_vars: []
        })
      ]
    })

    await renderMessaging('zh')

    fireEvent.change(await screen.findByLabelText('绝不回复'), {
      target: { value: '发票 + 逾期' }
    })
    fireEvent.change(screen.getByLabelText('必须回复'), {
      target: { value: '逾期 && 发票' }
    })
    fireEvent.click(screen.getByRole('button', { name: '保存更改' }))

    expect(updateMessagingPlatform).not.toHaveBeenCalled()
    expect(updateEmailAutoReplyPolicy).not.toHaveBeenCalled()
    expect(notifyError).toHaveBeenCalledWith(expect.any(Error), '关键词规则冲突')
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

    await waitFor(() => expect(approvePairing).toHaveBeenCalledWith('teams', 'a1b2c3d4e5f60718'))
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
