// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MessagingPlatformInfo } from '@/types/hermes'

const getMessagingPlatforms = vi.fn()
const updateMessagingPlatform = vi.fn()
const openExternalLink = vi.fn()
const notify = vi.fn()
const notifyError = vi.fn()
const runGatewayRestart = vi.fn()

vi.mock('@/hermes', () => ({
  getMessagingPlatforms: () => getMessagingPlatforms(),
  updateMessagingPlatform: (id: string, body: unknown) => updateMessagingPlatform(id, body)
}))

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

vi.mock('@/store/notifications', () => ({
  notify: (payload: unknown) => notify(payload),
  notifyError: (error: unknown, fallback: string) => notifyError(error, fallback)
}))

vi.mock('@/store/system-actions', () => ({
  runGatewayRestart: () => runGatewayRestart()
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
  runGatewayRestart.mockResolvedValue(undefined)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderMessaging(initialEntry = '/') {
  const { MessagingView } = await import('./index')
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <MemoryRouter initialEntries={[initialEntry]}>
        <MessagingView />
      </MemoryRouter>
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
})

describe('MessagingView restart affordance', () => {
  it('auto-restarts via runGatewayRestart after saving credentials', async () => {
    getMessagingPlatforms.mockResolvedValue({
      platforms: [
        platform({
          configured: true,
          enabled: true,
          env_vars: [
            {
              advanced: false,
              description: 'Discord bot token',
              is_password: true,
              is_set: false,
              key: 'DISCORD_BOT_TOKEN',
              prompt: 'Bot token',
              redacted_value: '',
              required: true,
              url: ''
            }
          ],
          gateway_running: false,
          id: 'discord',
          name: 'Discord',
          state: 'gateway_stopped'
        })
      ]
    })
    updateMessagingPlatform.mockResolvedValue({ ok: true, platform: 'discord' })

    await renderMessaging('/messaging?platform=discord')

    const input = await screen.findByLabelText('Bot token')
    await act(async () => {
      fireEvent.change(input, { target: { value: 'token-123' } })
    })

    const saveButton = screen.getByRole('button', { name: 'Save changes' })
    await act(async () => {
      fireEvent.click(saveButton)
    })

    await waitFor(() => {
      expect(updateMessagingPlatform).toHaveBeenCalledWith('discord', {
        env: { DISCORD_BOT_TOKEN: 'token-123' }
      })
    })

    await waitFor(() => expect(runGatewayRestart).toHaveBeenCalledTimes(1))

    const successPayload = notify.mock.calls
      .map(call => call[0] as { action?: unknown; message?: string; title?: string })
      .find(payload => payload?.title === 'Discord setup saved')

    expect(successPayload).toBeTruthy()
    expect(successPayload?.message).toBe('New credentials take effect after a gateway restart.')
    expect(successPayload?.action).toBeUndefined()
    expect(notifyError).not.toHaveBeenCalled()
  })
})
