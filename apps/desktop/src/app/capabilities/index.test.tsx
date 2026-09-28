// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import type * as ReactRouterDom from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type * as HubActions from '@/store/hub-actions'

import { parseCatalog } from './catalog/catalog-data'
import { $catalogCardView } from './catalog/store'

const getSkills = vi.fn()
const getToolsets = vi.fn()
const setSkillEnabled = vi.fn()
const setToolsetEnabled = vi.fn()
const getToolsetConfig = vi.fn()
const selectToolsetProvider = vi.fn()
const getUsageAnalytics = vi.fn()
const getProfiles = vi.fn()
const getSkillContent = vi.fn()
const getOfficialSkills = vi.fn()

// Partial mock: keep the real module (CapabilitiesView pulls in @/store/profile,
// whose import-time subscription calls setApiRequestProfile) and stub only the
// calls we assert on. Args are forwarded so the per-profile scope arg is
// observable.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getSkills: (profile?: null | string) => getSkills(profile),
  getToolsets: (profile?: null | string) => getToolsets(profile),
  setSkillEnabled: (name: string, enabled: boolean, profile?: null | string) => setSkillEnabled(name, enabled, profile),
  setToolsetEnabled: (name: string, enabled: boolean, profile?: null | string) =>
    setToolsetEnabled(name, enabled, profile),
  getToolsetConfig: (name: string, profile?: null | string) => getToolsetConfig(name, profile),
  selectToolsetProvider: (toolset: string, provider: string) => selectToolsetProvider(toolset, provider),
  getUsageAnalytics: (days: number, profile?: null | string) => getUsageAnalytics(days, profile),
  getProfiles: () => getProfiles(),
  getSkillContent: (name: string, profile?: null | string) => getSkillContent(name, profile),
  getOfficialSkills: (profile?: null | string) => getOfficialSkills(profile)
}))

// Notifications hit nanostores/timers we don't care about here.
vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

// The catalog Install button routes through the hub action pipeline — stub the
// action entrypoint (real module kept: CapabilitiesView reads $hubActions and the
// query keys from it).
vi.mock('@/store/hub-actions', async importOriginal => ({
  ...(await importOriginal<typeof HubActions>()),
  installHubSkill: vi.fn().mockResolvedValue(undefined)
}))

// The vision detail navigates to Settings → Models via useNavigate; spy on it
// so the deep-link target is assertable.
const navigateSpy = vi.fn()

vi.mock('react-router', async importOriginal => ({
  ...(await importOriginal<typeof ReactRouterDom>()),
  useNavigate: () => navigateSpy
}))

// Import at module scope (after the hoisted vi.mock calls) so the heavy
// component-tree transform is paid during collection, not billed against the
// first test's testTimeout — same flake class as messaging/index.test.tsx.
const { CapabilitiesView } = await import('./index')

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

async function renderSkills() {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      // CapabilitiesView reads skills/toolsets via useQuery, so it needs a provider.
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/capabilities?tab=toolsets']}>
          <CapabilitiesView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  $catalogCardView.set(true)
  getSkills.mockResolvedValue([])
  getToolsets.mockResolvedValue([toolset()])
  setToolsetEnabled.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: true, active_provider: null, providers: [] })
  getUsageAnalytics.mockResolvedValue({ tools: [] })
  getOfficialSkills.mockResolvedValue({ skills: [] })
  getSkillContent.mockResolvedValue({
    name: 'web-research',
    path: '/skills/web-research/SKILL.md',
    content: '---\nname: web-research\nversion: 1.2.0\nauthor: Nous\n---\n\n# Web Research\n\nDeep research steps.'
  })
  // Single profile by default → the scope selector stays hidden (>1 gate),
  // so existing tests see unchanged single-profile behavior.
  getProfiles.mockResolvedValue({ profiles: [{ name: 'default', is_default: true }] })
  queryClient.setQueryData(['public-catalog', 'skills'], [])
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  // Shared singleton client — drop cached skills/toolsets so each test refetches.
  queryClient.clear()
})

// CapabilitiesView is a heavy module (import cost now paid at module scope above,
// during collection) but the file still legitimately runs ~14s on CI runners —
// right against the global 15s per-test budget, so slow runners cascade-fail
// all 11 tests (2× in a row on PR #93612, plus a main run the same hour).
// Give this file headroom; the tests are not slow individually.
describe('CapabilitiesView toolset management', { timeout: 60_000 }, () => {
  it('renders a switch for each toolset and toggles it off', async () => {
    await renderSkills()

    // The switch names the action, so an enabled toolset offers to turn it off.
    const sw = await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })
    expect(sw.getAttribute('aria-checked')).toBe('true')

    await act(async () => {
      fireEvent.click(sw)
    })

    await waitFor(() => expect(setToolsetEnabled).toHaveBeenCalled())
    expect(setToolsetEnabled.mock.calls[0].slice(0, 2)).toEqual(['web', false])
  })

  it('scopes Tools config to the profile chosen in the selector', async () => {
    // Two profiles → the "Configuring:" selector renders. Picking a non-active
    // profile must re-fetch toolsets scoped to THAT profile.
    // jsdom's scrollIntoView is missing/non-functional; Radix Select calls it
    // on open. Force a stub so the dropdown can render in the test env.
    Element.prototype.scrollIntoView = vi.fn()
    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'researcher', is_default: false }
      ]
    })

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=toolsets']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    // The selector appears with >1 profile.
    const trigger = await screen.findByRole('combobox')
    await act(async () => {
      fireEvent.click(trigger)
    })
    const option = await screen.findByRole('option', { name: 'researcher' })
    await act(async () => {
      fireEvent.click(option)
    })

    // Toolsets refetch scoped to the picked profile.
    await waitFor(() => expect(getToolsets).toHaveBeenCalledWith('researcher'))
  })

  it('scopes the Skills tab (and skill toggles) to the profile chosen in the selector', async () => {
    // The selector is Capabilities-WIDE: picking a profile on the Skills tab
    // must refetch the skill list scoped to it, and route toggles there too.
    Element.prototype.scrollIntoView = vi.fn()
    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'researcher', is_default: false }
      ]
    })
    getSkills.mockResolvedValue([
      {
        name: 'web-research',
        description: 'Research the web',
        category: 'research',
        enabled: true,
        usage: 3,
        provenance: 'bundled'
      }
    ])

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    // The selector renders on the Skills tab too (Capabilities-wide), above the catalog's sort select.
    await waitFor(() => expect(screen.getAllByRole('combobox')).toHaveLength(2))
    const [trigger] = screen.getAllByRole('combobox')
    await act(async () => {
      fireEvent.click(trigger)
    })
    const option = await screen.findByRole('option', { name: 'researcher' })
    await act(async () => {
      fireEvent.click(option)
    })

    // Skills refetch scoped to the picked profile...
    await waitFor(() => expect(getSkills).toHaveBeenCalledWith('researcher'))

    // ...and a toggle routes its write to that profile as well.
    const sw = await screen.findByRole('switch', { name: 'web-research' })
    await act(async () => {
      fireEvent.click(sw)
    })
    await waitFor(() => expect(setSkillEnabled).toHaveBeenCalledWith('web-research', false, 'researcher'))
  })

  it('shows the FULL skill body in the detail pane, not just the description or raw frontmatter', async () => {
    $catalogCardView.set(false)
    getSkills.mockResolvedValue([
      {
        name: 'web-research',
        description: 'Research the web',
        category: 'research',
        enabled: true,
        usage: 3,
        provenance: 'bundled'
      }
    ])

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    await waitFor(() => expect(getSkillContent).toHaveBeenCalled())
    expect(getSkillContent.mock.calls[0][0]).toBe('web-research')
    expect(await screen.findByText(/Deep research steps/)).toBeTruthy()
    expect(screen.queryByText(/version: 1\.2\.0/)).toBeNull()
  })

  it('keeps installed skills on their toggle and installs new cards into the pinned remote profile', async () => {
    const { installHubSkill } = await import('@/store/hub-actions')
    getSkills.mockResolvedValue([
      { name: 'web-research', description: 'Research', enabled: true, category: 'research' }
    ])
    queryClient.setQueryData(
      ['public-catalog', 'skills'],
      parseCatalog('skills', [
        { name: 'web-research', source: 'official', identifier: 'official/research/web-research' },
        { name: 'gif-search', source: 'official', identifier: 'official/gifs/gif-search' }
      ])
    )

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <CapabilitiesView embedded fixedConnection="homelab" fixedProfile="inbox-bot" />
        </MemoryRouter>
      </QueryClientProvider>
    )

    await screen.findByRole('switch', { name: 'web-research' })
    expect(screen.getByRole<HTMLButtonElement>('switch', { name: 'Added web-research' }).disabled).toBe(true)
    expect(screen.queryByRole('switch', { name: 'Add web-research' })).toBeNull()
    const available = screen.getByRole('button', { name: 'gif-search' }).closest('article')!
    fireEvent.click(within(available).getByRole('switch', { name: 'Add gif-search' }))
    await waitFor(() =>
      expect(installHubSkill).toHaveBeenCalledExactlyOnceWith('official/gifs/gif-search', {
        connectionId: 'homelab',
        profile: 'inbox-bot'
      })
    )
    expect(document.querySelector('iframe')).toBeNull()
  })

  it('fetches the public catalog once across remounts and keeps one search input', async () => {
    queryClient.clear()

    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [{ name: 'example', source: 'official', identifier: 'official/example' }]
    })

    vi.stubGlobal('fetch', fetch)

    const view = () =>
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter>
            <CapabilitiesView embedded fixedProfile="default" />
          </MemoryRouter>
        </QueryClientProvider>
      )

    try {
      const first = view()
      await screen.findByRole('button', { name: 'example' })
      expect(fetch).toHaveBeenCalledTimes(1)
      first.unmount()
      view()
      await screen.findByRole('button', { name: 'example' })
      expect(fetch).toHaveBeenCalledTimes(1)
      expect(screen.getAllByRole('textbox', { name: 'Search skills' })).toHaveLength(1)
      expect(document.querySelector('iframe')).toBeNull()
    } finally {
      vi.unstubAllGlobals()
    }
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

    const link = await screen.findByRole('button', { name: /Choose vision model in Settings/ })

    await act(async () => {
      fireEvent.click(link)
    })

    // Internal route change into the Models section with the aux slot target —
    // consumed by ModelSettings' deep-link highlight. Never an external URL.
    await waitFor(() => expect(navigateSpy).toHaveBeenCalledWith('/settings?tab=config:model&aux=vision'))
  })

  it('fixedConnection pins every read to the target connection', async () => {
    // Bot Mode's remote-target door: a bot on another registered gateway gets
    // the live surface pointed at ITS backend — the reads must carry the
    // (connection, profile) pin, not a bare profile name that would resolve
    // against the ACTIVE gateway (the wrong-machine bug).
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities']}>
            <CapabilitiesView embedded fixedConnection="homelab" fixedProfile="inbox-bot" />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    await waitFor(() => expect(getSkills).toHaveBeenCalled())
    expect(getSkills.mock.calls[0][0]).toEqual({ connectionId: 'homelab', profile: 'inbox-bot' })
    expect(getToolsets.mock.calls[0][0]).toEqual({ connectionId: 'homelab', profile: 'inbox-bot' })
    // Pinned scope → no roster/profiles fetch, selector hidden.
    expect(getProfiles).not.toHaveBeenCalled()
  })

  it('offers (connection, profile) scope rows on multi-connection desktops', async () => {
    // With a v2 registry holding >1 connection, the scope selector lists the
    // union agent roster — profile + owning device — instead of the local
    // profiles list, so a selection identifies WHICH gateway's capabilities
    // are being configured.
    const connections = {
      list: vi.fn().mockResolvedValue({
        version: 2,
        primary: 'local',
        secureTokenStorage: true,
        connections: [
          { id: 'local', kind: 'local', label: 'This device', tokenSet: false, tokenPreview: null },
          { id: 'homelab', kind: 'remote', label: 'Homelab', tokenSet: true, tokenPreview: '…' }
        ]
      })
    }

    const getAgentRoster = vi.fn().mockResolvedValue({
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local',
          connectionLabel: 'This device',
          profile: 'default',
          handle: 'default'
        },
        {
          connectionId: 'homelab',
          connectionKind: 'remote',
          connectionLabel: 'Homelab',
          profile: 'inbox-bot',
          handle: 'inbox-bot-homelab'
        }
      ],
      sources: []
    })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { connections, getAgentRoster }

    try {
      await renderSkills()

      await waitFor(() => expect(getAgentRoster).toHaveBeenCalled())
      // The selector paints roster rows labeled profile — device.
      expect(await screen.findByText('default — This device (current)')).toBeTruthy()
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('lists the built-in optional-skills catalog with Add switches that route through the hub pipeline', async () => {
    // Official optional skills merge into the same catalog as installed ones;
    // an uninstalled row carries an Add switch that routes through the hub
    // action pipeline scoped to the Capabilities profile. A name collision with
    // an installed skill keeps the installed toggle, never a second Add.
    const { installHubSkill } = await import('@/store/hub-actions')

    getSkills.mockResolvedValue([
      {
        name: 'web-research',
        description: 'Research the web',
        category: 'research',
        enabled: true,
        usage: 3,
        provenance: 'bundled'
      }
    ])
    getOfficialSkills.mockResolvedValue({
      skills: [
        {
          name: 'gif-search',
          description: 'Search GIFs',
          identifier: 'official/gifs/gif-search',
          category: 'gifs',
          installed: false,
          tags: ['gifs']
        },
        {
          name: 'web-research',
          description: 'already here under a different source',
          identifier: 'official/research/web-research',
          category: 'research',
          installed: false,
          tags: []
        },
        {
          name: 'ascii-art',
          description: 'ASCII art',
          identifier: 'official/creative/ascii-art',
          category: 'creative',
          installed: true,
          tags: []
        }
      ]
    })

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    const install = await screen.findByRole('switch', { name: 'Add gif-search' })
    expect(screen.getByRole('switch', { name: 'web-research' })).toBeTruthy()
    expect(screen.queryByRole('switch', { name: 'Add web-research' })).toBeNull()
    expect(screen.getByRole<HTMLButtonElement>('switch', { name: 'Added ascii-art' }).disabled).toBe(true)

    await act(async () => {
      fireEvent.click(install)
    })

    await waitFor(() =>
      expect(vi.mocked(installHubSkill)).toHaveBeenCalledWith('official/gifs/gif-search', expect.anything())
    )
  })
})
