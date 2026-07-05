// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MessagingPlatformInfo, ProfileInfo } from '@/types/hermes'

const getMessagingPlatforms = vi.fn()
const updateMessagingPlatform = vi.fn()
const getProfiles = vi.fn()
const openExternalLink = vi.fn()

vi.mock('@/hermes', () => ({
  // getProfiles/setApiRequestProfile are required so @/store/profile's
  // module-level $activeGatewayProfile subscription (imported by
  // messaging/index.tsx for the cross-profile rail) doesn't crash without
  // window.hermesDesktop — see profile.test.ts for the same pattern.
  getProfiles: () => getProfiles(),
  setApiRequestProfile: vi.fn(),
  getMessagingPlatforms: (profile?: null | string) => getMessagingPlatforms(profile),
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

function profileInfo(patch: Partial<ProfileInfo> = {}): ProfileInfo {
  return {
    gateway_running: true,
    has_env: false,
    is_default: false,
    model: null,
    name: 'default',
    path: '/tmp/hermes',
    provider: null,
    skill_count: 0,
    ...patch
  }
}

beforeEach(() => {
  updateMessagingPlatform.mockResolvedValue({ ok: true, platform: 'teams' })
  // Single-profile baseline: no rail, same behavior as before this feature.
  getProfiles.mockResolvedValue({ profiles: [] })
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

describe('MessagingView cross-profile rail', () => {
  it('does not render a rail with a single profile (default, unaffected behavior)', async () => {
    getProfiles.mockResolvedValue({ profiles: [profileInfo({ name: 'default' })] })
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform()] })

    await renderMessaging()

    await screen.findAllByText('Microsoft Teams')
    expect(screen.queryByRole('button', { name: 'default' })).toBeNull()
    // Resolved to the concrete profile name, never an ambiguous null/current —
    // see the regression test below for why that distinction matters.
    expect(getMessagingPlatforms).toHaveBeenCalledWith('default')
  })

  it('re-scopes the page to another profile on click, without switching the app-wide active profile', async () => {
    getProfiles.mockResolvedValue({
      profiles: [
        profileInfo({ name: 'default', is_default: true, gateway_running: true }),
        profileInfo({ name: 'health', gateway_running: false })
      ]
    })
    getMessagingPlatforms.mockImplementation(async (profile?: null | string) =>
      profile === 'health'
        ? { platforms: [platform({ enabled: true, id: 'wechat', name: 'WeChat', state: 'fatal' })] }
        : { platforms: [platform()] }
    )

    await renderMessaging()

    await screen.findAllByText('Microsoft Teams')
    expect(getMessagingPlatforms).toHaveBeenCalledWith('default')

    fireEvent.click(await screen.findByRole('button', { name: 'health' }))

    await waitFor(() => expect(getMessagingPlatforms).toHaveBeenCalledWith('health'))
    expect((await screen.findAllByText('WeChat')).length).toBeGreaterThan(0)
  })

  it('regression: writes always carry the resolved profile, never an ambiguous null/current', async () => {
    // _profile_scope(None/""/"current") on the backend resolves to whatever
    // profile the PRIMARY BACKEND PROCESS happens to be running as — which
    // can drift from $activeGatewayProfile (e.g. a chat-session gateway
    // swap moves it without restarting the primary backend). Toggling a
    // platform while viewing "default" must send profile: 'default'
    // explicitly, or the write can silently land in whatever profile the
    // backend process itself is bound to instead (#59011 follow-up).
    getProfiles.mockResolvedValue({
      profiles: [profileInfo({ name: 'default' }), profileInfo({ name: 'health' })]
    })
    getMessagingPlatforms.mockResolvedValue({ platforms: [platform({ enabled: false })] })

    await renderMessaging()
    await screen.findAllByText('Microsoft Teams')

    const toggle = await screen.findByRole('switch')
    fireEvent.click(toggle)

    await waitFor(() =>
      expect(updateMessagingPlatform).toHaveBeenCalledWith('teams', { enabled: true, profile: 'default' })
    )
  })
})
