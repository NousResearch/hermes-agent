// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import type * as ReactRouterDom from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'

import { ChannelsTab } from './channels-tab'

const getSkills = vi.fn()
const getToolsets = vi.fn()
const setSkillEnabled = vi.fn()
const setToolsetEnabled = vi.fn()
const getToolsetConfig = vi.fn()
const selectToolsetProvider = vi.fn()
const getUsageAnalytics = vi.fn()
const getChannelCapabilities = vi.fn()
const updateChannelCapabilities = vi.fn()

const { notify, notifyError } = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

// Partial mock: keep the real module (SkillsView pulls in @/store/profile,
// whose import-time subscription calls setApiRequestProfile) and stub only the
// calls we assert on.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getSkills: () => getSkills(),
  getToolsets: () => getToolsets(),
  setSkillEnabled: (name: string, enabled: boolean) => setSkillEnabled(name, enabled),
  setToolsetEnabled: (name: string, enabled: boolean) => setToolsetEnabled(name, enabled),
  getToolsetConfig: (name: string) => getToolsetConfig(name),
  selectToolsetProvider: (toolset: string, provider: string) => selectToolsetProvider(toolset, provider),
  getUsageAnalytics: (days: number) => getUsageAnalytics(days),
  getChannelCapabilities: () => getChannelCapabilities(),
  updateChannelCapabilities: (platform: string, update: unknown) =>
    updateChannelCapabilities(platform, update)
}))

// Notifications hit nanostores/timers we don't care about here.
vi.mock('@/store/notifications', () => ({
  notify,
  notifyError
}))

// The vision detail navigates to Settings → Models via useNavigate; spy on it
// so the deep-link target is assertable.
const navigateSpy = vi.fn()

vi.mock('react-router', async importOriginal => ({
  ...(await importOriginal<typeof ReactRouterDom>()),
  useNavigate: () => navigateSpy
}))

function toolset(overrides: Record<string, unknown> = {}) {
  return {
    name: 'web',
    label: 'Web Search',
    description: 'web_search, web_extract',
    enabled: true,
    available: true,
    configured: true,
    tools: ['web_search', 'web_extract'],
    ...overrides
  }
}

function channelCapability(overrides: Record<string, unknown> = {}) {
  return {
    effective_toolsets: ['web'],
    explicit: true,
    implicit_toolsets: [],
    label: 'Email',
    mcp: {
      available: ['alpha'],
      effective: ['alpha'],
      mode: 'allowlist' as const,
      selected: ['alpha']
    },
    platform: 'email',
    plugins_locked: false,
    toolsets: [
      {
        description: 'Search the web',
        enabled: true,
        label: 'Web',
        name: 'web',
        tools: ['web_search']
      },
      {
        description: 'Remember context',
        enabled: false,
        label: 'Memory',
        name: 'memory',
        tools: ['memory_search']
      }
    ],
    ...overrides
  }
}

async function renderSkills() {
  const { SkillsView } = await import('./index')
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      // SkillsView reads skills/toolsets via useQuery, so it needs a provider.
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/skills?tab=toolsets']}>
          <SkillsView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

async function renderChannels(query = '') {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <QueryClientProvider client={queryClient}>
        <ChannelsTab query={query} />
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  $activeGatewayProfile.set('default')
  getSkills.mockResolvedValue([])
  getToolsets.mockResolvedValue([toolset()])
  setToolsetEnabled.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: true, active_provider: null, providers: [] })
  getUsageAnalytics.mockResolvedValue({ tools: [] })
  getChannelCapabilities.mockResolvedValue([channelCapability()])
  updateChannelCapabilities.mockResolvedValue({ ok: true, channel: channelCapability() })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $activeGatewayProfile.set('default')
  // Shared singleton client — drop cached skills/toolsets so each test refetches.
  queryClient.clear()
})

describe('SkillsView toolset management', () => {
  it('renders a switch for each toolset and toggles it off', async () => {
    await renderSkills()

    // The switch names the action, so an enabled toolset offers to turn it off.
    const sw = await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })
    expect(sw.getAttribute('aria-checked')).toBe('true')

    await act(async () => {
      fireEvent.click(sw)
    })

    await waitFor(() => expect(setToolsetEnabled).toHaveBeenCalledWith('web', false))
  })

  it('renders toolset titles without leading emoji', async () => {
    getToolsets.mockResolvedValue([toolset({ name: 'cronjob', label: '⏰ Cron Jobs', description: 'cron tools' })])

    await renderSkills()

    // The label renders in both the row and the auto-selected detail header, so
    // assert via the switch's (emoji-stripped) accessible name and the absence
    // of the emoji rather than a single-match text lookup.
    await screen.findByRole('switch', { name: 'Turn Cron Jobs toolset off' })
    expect(screen.queryByText(/⏰/)).toBeNull()
  })

  it('renders the provider config panel inline for the selected toolset', async () => {
    // The master-detail UI dropped the resting "Configured" pill and the
    // "Configure" expander: the detail column auto-selects the first toolset
    // and renders its config panel directly, which fetches on mount.
    await renderSkills()

    await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })
    await waitFor(() => expect(getToolsetConfig).toHaveBeenCalledWith('web'))
  })

  it('shows a vision explainer that deep-links to Settings → Models', async () => {
    // Vision has no TOOL_CATEGORIES provider matrix — its model lives in the
    // auxiliary model config, so the detail pane must point there instead of
    // rendering an empty panel.
    getToolsets.mockResolvedValue([
      toolset({
        name: 'vision',
        label: 'Vision / Image Analysis',
        description: 'vision_analyze',
        tools: ['vision_analyze']
      })
    ])
    getToolsetConfig.mockResolvedValue({ has_category: false, active_provider: null, providers: [] })

    await renderSkills()

    expect(await screen.findByText(/auxiliary model configuration/)).toBeTruthy()
    const link = screen.getByRole('button', { name: /Choose vision model in Settings/ })

    await act(async () => {
      fireEvent.click(link)
    })

    // Internal route change into the Models section with the aux slot target —
    // consumed by ModelSettings' deep-link highlight. Never an external URL.
    await waitFor(() => expect(navigateSpy).toHaveBeenCalledWith('/settings?tab=config:model&aux=vision'))
  })
})

describe('ChannelsTab channel boundaries', () => {
  it('keeps an unsaved draft while filtering and saves the exact MCP boundary', async () => {
    getChannelCapabilities.mockResolvedValue([
      channelCapability(),
      channelCapability({
        label: 'Telegram',
        platform: 'telegram'
      })
    ])

    const result = await renderChannels()

    const memory = await screen.findByRole('switch', {
      name: 'Toggle Memory for this channel'
    })

    fireEvent.click(memory)

    result.rerender(
      <QueryClientProvider client={queryClient}>
        <ChannelsTab query="Telegram" />
      </QueryClientProvider>
    )
    expect(screen.getByRole('switch', { name: 'Toggle Memory for this channel' }).getAttribute('aria-checked')).toBe(
      'true'
    )

    fireEvent.click(screen.getByRole('button', { name: 'Save abilities' }))
    await waitFor(() =>
      expect(updateChannelCapabilities).toHaveBeenCalledWith('email', {
        toolsets: ['memory', 'web'],
        mcp_mode: 'allowlist',
        mcp_servers: ['alpha']
      })
    )
  })

  it('locks channel selection while the current channel boundary is saving', async () => {
    getChannelCapabilities.mockResolvedValue([
      channelCapability(),
      channelCapability({
        label: 'Telegram',
        platform: 'telegram'
      })
    ])
    let finishSave: (() => void) | undefined
    updateChannelCapabilities.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          finishSave = resolve
        })
    )

    await renderChannels()

    fireEvent.click(await screen.findByRole('button', { name: 'Save abilities' }))
    await waitFor(() => expect(updateChannelCapabilities).toHaveBeenCalledOnce())

    expect(screen.getByRole('button', { name: 'Email' })).toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: 'Telegram' })).toHaveProperty('disabled', true)

    finishSave?.()
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Telegram' })).toHaveProperty('disabled', false)
    )
  })

  it('ignores a prior profile save while the active profile is saving', async () => {
    const primary = channelCapability({ label: 'Primary Email', platform: 'email' })
    const review = channelCapability({ label: 'Review Telegram', platform: 'telegram' })
    getChannelCapabilities.mockImplementation(() =>
      Promise.resolve($activeGatewayProfile.get() === 'review' ? [review] : [primary])
    )

    let finishPrimary: (() => void) | undefined
    let finishReview: (() => void) | undefined
    updateChannelCapabilities.mockImplementation(platform =>
      new Promise<void>(resolve => {
        if (platform === 'email') {
          finishPrimary = resolve
        } else {
          finishReview = resolve
        }
      })
    )

    await renderChannels()
    fireEvent.click(await screen.findByRole('button', { name: 'Save abilities' }))
    await waitFor(() => expect(updateChannelCapabilities).toHaveBeenCalledTimes(1))

    await act(async () => {
      $activeGatewayProfile.set('review')
    })
    await screen.findByRole('button', { name: 'Review Telegram' })

    const reviewSave = screen.getByRole('button', { name: 'Save abilities' })
    expect(reviewSave).toHaveProperty('disabled', false)
    fireEvent.click(reviewSave)
    await waitFor(() => expect(updateChannelCapabilities).toHaveBeenCalledTimes(2))

    finishPrimary?.()
    await waitFor(() => expect(reviewSave).toHaveProperty('disabled', true))
    expect(screen.getByRole('button', { name: 'Review Telegram' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Primary Email' })).toBeNull()
    expect(notify).not.toHaveBeenCalled()

    finishReview?.()
    await waitFor(() => expect(reviewSave).toHaveProperty('disabled', false))
  })
})
