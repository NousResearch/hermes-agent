// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MessagingPlatformInfo } from '@/types/hermes'

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

beforeEach(() => {
  updateMessagingPlatform.mockResolvedValue({ ok: true, platform: 'teams' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderMessaging() {
  const { MessagingView } = await import('./index')

  return render(
    <MemoryRouter>
      <MessagingView />
    </MemoryRouter>
  )
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
    fireEvent.click(link)

    await waitFor(() => expect(openExternalLink).toHaveBeenCalledWith(docsUrl))
  })
})

describe('MessagingView option fields', () => {
  it('uses a segmented mode control and hides cloud-only Photon credentials in local mode', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          id: 'photon',
          name: 'iMessage via Photon',
          env_vars: [
            {
              advanced: false,
              default_value: 'cloud',
              description: 'Choose the iMessage connection.',
              is_password: false,
              is_set: false,
              key: 'PHOTON_IMESSAGE_MODE',
              options: [
                { label: 'Photon cloud', value: 'cloud' },
                { label: 'Local Mac', value: 'local' }
              ],
              prompt: 'iMessage connection',
              redacted_value: null,
              required: false,
              url: null,
              value: null,
              visible_when: null
            },
            {
              advanced: false,
              description: 'Cloud project id.',
              is_password: false,
              is_set: false,
              key: 'PHOTON_PROJECT_ID',
              prompt: 'Photon Spectrum project id',
              redacted_value: null,
              required: true,
              url: null,
              visible_when: { key: 'PHOTON_IMESSAGE_MODE', values: ['cloud'] }
            },
            {
              advanced: false,
              description: 'Cloud project secret.',
              is_password: true,
              is_set: false,
              key: 'PHOTON_PROJECT_SECRET',
              prompt: 'Photon project secret',
              redacted_value: null,
              required: true,
              url: null,
              visible_when: { key: 'PHOTON_IMESSAGE_MODE', values: ['cloud'] }
            }
          ]
        })
      ]
    })

    await renderMessaging()

    expect((await screen.findByRole('button', { name: 'Photon cloud' })).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('Photon Spectrum project id')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Local Mac' }))

    expect(screen.getByRole('button', { name: 'Local Mac' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.queryByText('Photon Spectrum project id')).toBeNull()
    expect(screen.queryByText('Photon project secret')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Photon cloud' }))
    expect(screen.getByText('Photon Spectrum project id')).toBeTruthy()
    expect(screen.getByText('Photon project secret')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Local Mac' }))

    fireEvent.click(screen.getByRole('button', { name: /save changes/i }))

    await waitFor(() =>
      expect(updateMessagingPlatform).toHaveBeenCalledWith('photon', {
        env: { PHOTON_IMESSAGE_MODE: 'local' }
      })
    )
  })
})
