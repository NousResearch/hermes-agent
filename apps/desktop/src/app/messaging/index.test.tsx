// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { MessagingEnvVarInfo, MessagingPlatformInfo } from '@/types/hermes'

const getMessagingPlatforms = vi.fn()
const updateMessagingPlatform = vi.fn()
const openExternalLink = vi.fn()

vi.mock('@/hermes', () => ({
  getMessagingPlatforms: () => getMessagingPlatforms(),
  updateMessagingPlatform: (id: string, body: unknown) => updateMessagingPlatform(id, body)
}))

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
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

function envVar(patch: Partial<MessagingEnvVarInfo> = {}): MessagingEnvVarInfo {
  return {
    advanced: false,
    description: 'A field.',
    is_password: false,
    is_set: false,
    key: 'EXAMPLE_KEY',
    prompt: 'Example field',
    redacted_value: null,
    required: true,
    url: null,
    ...patch
  }
}

beforeEach(() => {
  updateMessagingPlatform.mockResolvedValue({ ok: true, platform: 'teams' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderMessaging(locale: 'ar' | 'en' = 'en') {
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

  it('localizes required Mattermost credential help in Arabic', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          id: 'mattermost',
          name: 'Mattermost',
          description: 'Connect Hermes to Mattermost channels and direct messages.',
          env_vars: [
            envVar({
              key: 'MATTERMOST_URL',
              prompt: 'Mattermost server URL',
              description: 'Mattermost server URL (e.g. https://mm.example.com)'
            }),
            envVar({
              key: 'MATTERMOST_TOKEN',
              prompt: 'Mattermost bot token',
              description: 'Mattermost bot token or personal access token',
              is_password: true
            })
          ]
        })
      ]
    })

    await renderMessaging('ar')

    expect(await screen.findByText('رابط خادم ماترموست، مثل الرابط الذي يبدأ ببروتوكول الاتصال الآمن.')).toBeTruthy()
    expect(screen.getByText('رمز بوت ماترموست أو رمز وصول شخصي.')).toBeTruthy()
    expect(screen.queryByText(/Mattermost server URL/)).toBeNull()
  })

  it('localizes required Matrix credential help in Arabic', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          id: 'matrix',
          name: 'Matrix',
          description: 'Use Hermes in Matrix rooms and direct messages.',
          env_vars: [
            envVar({
              key: 'MATRIX_HOMESERVER',
              prompt: 'Matrix homeserver URL',
              description: 'Matrix homeserver URL (e.g. https://matrix.example.org)'
            }),
            envVar({
              key: 'MATRIX_ACCESS_TOKEN',
              prompt: 'Matrix access token',
              description: 'Matrix access token (preferred over password login)',
              is_password: true
            }),
            envVar({
              key: 'MATRIX_USER_ID',
              prompt: 'Matrix user ID (@user:server)',
              description: 'Matrix user ID (e.g. @hermes:example.org)'
            })
          ]
        })
      ]
    })

    await renderMessaging('ar')

    expect(await screen.findByText('رابط الخادم الرئيسي لشبكة ماتريكس.')).toBeTruthy()
    expect(screen.getByText('رمز وصول ماتريكس، وهو مفضّل على تسجيل الدخول بكلمة المرور.')).toBeTruthy()
    expect(screen.getByText('معرّف مستخدم ماتريكس الكامل للبوت.')).toBeTruthy()
    expect(screen.queryByText(/Matrix homeserver URL/)).toBeNull()
  })

  it('localizes the Google Chat setup introduction in Arabic', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          id: 'google_chat',
          name: 'Google Chat',
          description: 'Connect Hermes to Google Chat via Cloud Pub/Sub.'
        })
      ]
    })

    await renderMessaging('ar')

    expect(await screen.findByText('اربط هرمس بمحادثات Google عبر خدمة النشر والاشتراك السحابية.')).toBeTruthy()
    expect(screen.queryByText('Connect Hermes to Google Chat via Cloud Pub/Sub.')).toBeNull()
  })
})
