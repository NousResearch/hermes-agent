// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { MemoryRouter } from 'react-router'
import type * as ReactRouterDom from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { TRANSLATIONS } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import type * as AgentPlugins from '@/store/agent-plugins'
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
const getHermesConfigRecord = vi.fn()
const loadAgentPlugins = vi.fn()

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
  getOfficialSkills: (profile?: null | string) => getOfficialSkills(profile),
  getHermesConfigRecord: (profile?: unknown) => getHermesConfigRecord(profile)
}))

// The plugin list load is the observable edge of the Plugins tab and its count.
vi.mock('@/store/agent-plugins', async importOriginal => ({
  ...(await importOriginal<typeof AgentPlugins>()),
  loadAgentPlugins: (request: unknown, profile?: null | string) => loadAgentPlugins(request, profile)
}))

// Notifications hit nanostores/timers we don't care about here.
// The tab's own data is not this file's subject: what the page hands it is.
vi.mock('./connectors/connectors-tab', () => ({
  ConnectorsTab: (props: { agentControls?: boolean; ownerLabel?: string; profile: unknown }) => (
    <div
      data-controls={String(props.agentControls ?? true)}
      data-owner={props.ownerLabel ?? ''}
      data-profile={JSON.stringify(props.profile ?? 'default')}
      data-testid="connectors-tab"
    />
  )
}))

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
const { $shellChoice } = await import('@/store/shell-variant')
const { $gatewayState } = await import('@/store/session')
const { SettingsOwnerContext } = await import('../settings/profile-scope')

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

async function renderSkills(fixedProfile?: string, fixedConnection?: string) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      // CapabilitiesView reads skills/toolsets via useQuery, so it needs a provider.
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/capabilities?tab=toolsets']}>
          <CapabilitiesView fixedConnection={fixedConnection} fixedProfile={fixedProfile} />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  $shellChoice.set(true)
  $catalogCardView.set(true)
  getSkills.mockResolvedValue([])
  getToolsets.mockResolvedValue([toolset()])
  setToolsetEnabled.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: true, active_provider: null, providers: [] })
  getUsageAnalytics.mockResolvedValue({ tools: [] })
  getOfficialSkills.mockResolvedValue({ skills: [] })
  getHermesConfigRecord.mockResolvedValue({})
  loadAgentPlugins.mockResolvedValue(true)
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
  $shellChoice.set(null)
  $gatewayState.set('idle')
  vi.clearAllMocks()
  // Shared singleton client — drop cached skills/toolsets so each test refetches.
  queryClient.clear()
})

// CapabilitiesView is a heavy module (import cost now paid at module scope above,
// during collection) but the file still legitimately runs ~14s on CI runners —
// right against the global 15s per-test budget, so slow runners cascade-fail
// all 11 tests (2× in a row on PR #93612, plus a main run the same hour).
// Give this file headroom; the tests are not slow individually.
describe('CapabilitiesView tabs', { timeout: 60_000 }, () => {
  async function tabsAt(path: string) {
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={[path]}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    const tabs = await screen.findAllByRole('tab')

    return {
      order: tabs.map(tab => tab.textContent?.replace(/\d+$/, '')),
      selected: tabs.find(tab => tab.getAttribute('aria-selected') === 'true')?.textContent?.replace(/\d+$/, '')
    }
  }

  it('leads with Connections in the redesign, and opens there', async () => {
    expect(await tabsAt('/capabilities')).toEqual({
      order: ['Connections', 'Plugins', 'Skills'],
      selected: 'Connections'
    })
  })

  it('still lands on the tab a link names', async () => {
    expect((await tabsAt('/capabilities?tab=skills')).selected).toBe('Skills')
  })

  it('has no Tools tab in the redesign, and an old link to it lands on the first tab', async () => {
    expect(await tabsAt('/capabilities?tab=toolsets')).toEqual({
      order: ['Connections', 'Plugins', 'Skills'],
      selected: 'Connections'
    })
  })

  it.each([true, false])('opens on the agent a link names (redesign %s)', async redesign => {
    $shellChoice.set(redesign)
    getToolsets.mockClear()
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=connectors&agentProfile=ops']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    await waitFor(() => expect(getToolsets).toHaveBeenCalled())
    expect(getToolsets.mock.calls.some(call => JSON.stringify(call).includes('ops'))).toBe(true)
  })

  it.each([
    [true, 'Agent settings hand-off gone'],
    [false, 'classic honors the linked agent']
  ])('manages connectors on the page itself, for the agent chosen (redesign %s: %s)', async redesign => {
    $shellChoice.set(redesign)
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=connectors&agentProfile=ops']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    const tab = await screen.findByTestId('connectors-tab')

    // The switches, "for me" and tool rules all hang off this one flag (connectors-tab.tsx).
    expect(tab.dataset.controls).toBe('true')
    expect(tab.dataset.profile).toBe('"ops"')
    expect(Boolean(tab.dataset.owner)).toBe(redesign)
    expect(screen.queryByRole('button', { name: TRANSLATIONS.en.skills.openAgentSettings })).toBeNull()
  })

  it('hands off to the agent\'s settings from Skills, where its switches are', async () => {
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=skills&agentProfile=ops']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    fireEvent.click(await screen.findByRole('button', { name: TRANSLATIONS.en.skills.openAgentSettings }))

    expect(navigateSpy).toHaveBeenCalledWith(expect.stringContaining('agentProfile=ops&panel=capabilities&tab=skills'))
  })

  describe('Plugins for one agent', () => {
    const library = { name: 'weather-plugin', description: 'Forecasts', source: 'official', category: 'weather' }
    const git = TRANSLATIONS.en.settings.plugins.installModal.installFromGit

    const pinned = async (props: ComponentProps<typeof CapabilitiesView>) => {
      queryClient.setQueryData(['public-catalog', 'plugins'], parseCatalog('plugins', [library]))
      await act(async () => {
        render(
          <QueryClientProvider client={queryClient}>
            <MemoryRouter initialEntries={['/settings?tab=plugins']}>
              <CapabilitiesView embedded fixedProfile="ops" {...props} />
            </MemoryRouter>
          </QueryClientProvider>
        )
      })
    }

    it('the agent\'s settings page lists only what is installed: no library, no way to install', async () => {
      await pinned({ fixedMode: 'plugins', installedPluginsOnly: true })

      await screen.findByRole('link', { name: 'Browse the plugin catalog' })
      expect(screen.queryByText('weather-plugin')).toBeNull()
      expect(screen.queryByRole('button', { name: git })).toBeNull()
    })

    it('a dialog that pins an agent keeps the library and Install from Git', async () => {
      await pinned({})
      // The dialog has the view's own tabs; the settings page pins one.
      fireEvent.click(document.querySelector<HTMLElement>('[data-tour="tab-plugins"]')!)

      expect(await screen.findByText('weather-plugin')).toBeTruthy()
      expect(screen.getByRole('button', { name: git })).toBeTruthy()
    })
  })

  it('holds the Connectors controls until another machine\'s roster says which profile serves the agent', async () => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      connections: {
        list: vi.fn().mockResolvedValue({
          version: 2,
          primary: 'local',
          secureTokenStorage: true,
          connections: [
            { id: 'local', kind: 'local', label: 'This device', tokenSet: false, tokenPreview: null },
            { id: 'homelab', kind: 'remote', label: 'Homelab', tokenSet: true, tokenPreview: '…' }
          ]
        })
      },
      getAgentRoster: vi.fn(() => new Promise(() => {}))
    }

    try {
      await act(async () => {
        render(
          <QueryClientProvider client={queryClient}>
            <MemoryRouter initialEntries={['/capabilities?tab=connectors&agentProfile=inbox-bot&connection=homelab']}>
              <CapabilitiesView />
            </MemoryRouter>
          </QueryClientProvider>
        )
      })

      expect((await screen.findByTestId('connectors-tab')).dataset.controls).toBe('false')
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('releases the Connectors controls once the roster answers', async () => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      connections: {
        list: vi.fn().mockResolvedValue({
          version: 2,
          primary: 'local',
          secureTokenStorage: true,
          connections: [
            { id: 'local', kind: 'local', label: 'This device', tokenSet: false, tokenPreview: null },
            { id: 'homelab', kind: 'remote', label: 'Homelab', tokenSet: true, tokenPreview: '…' }
          ]
        })
      },
      getAgentRoster: vi.fn().mockResolvedValue({
        agents: [{ connectionId: 'homelab', connectionLabel: 'Homelab', profile: 'inbox-bot' }]
      })
    }

    try {
      await act(async () => {
        render(
          <QueryClientProvider client={queryClient}>
            <MemoryRouter initialEntries={['/capabilities?tab=connectors&agentProfile=inbox-bot&connection=homelab']}>
              <CapabilitiesView />
            </MemoryRouter>
          </QueryClientProvider>
        )
      })

      await waitFor(() => expect(screen.getByTestId('connectors-tab').dataset.controls).toBe('true'))
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('does not hold the Connectors controls when only one connection is registered', async () => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      connections: {
        list: vi.fn().mockResolvedValue({
          version: 2,
          primary: 'local',
          secureTokenStorage: true,
          connections: [{ id: 'local', kind: 'local', label: 'This device', tokenSet: false, tokenPreview: null }]
        })
      },
      // A single-connection setup never fetches the union roster: a promise that never settles must not
      // keep the controls held.
      getAgentRoster: vi.fn(() => new Promise(() => {}))
    }

    try {
      await act(async () => {
        render(
          <QueryClientProvider client={queryClient}>
            <MemoryRouter initialEntries={['/capabilities?tab=connectors&agentProfile=inbox-bot&connection=local']}>
              <CapabilitiesView />
            </MemoryRouter>
          </QueryClientProvider>
        )
      })

      await waitFor(() => expect(screen.getByTestId('connectors-tab').dataset.controls).toBe('true'))
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('keeps classic order and its Skills default', async () => {
    $shellChoice.set(false)
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    // Classic's row is plain buttons, named for a tour by tab id.
    const tabs = [...document.querySelectorAll<HTMLElement>('[data-tour^="tab-"]')]

    expect(tabs.map(tab => tab.dataset.tour)).toEqual(['tab-skills', 'tab-plugins', 'tab-connectors', 'tab-toolsets'])
    expect(tabs.filter(tab => tab.dataset.active === 'true').map(tab => tab.dataset.tour)).toEqual(['tab-skills'])
  })
})

describe('CapabilitiesView toolset management', { timeout: 60_000 }, () => {
  it('renders when the browser has no idle callback APIs', async () => {
    vi.stubGlobal('requestIdleCallback', undefined)
    vi.stubGlobal('cancelIdleCallback', undefined)
    const prefetchSpy = vi.spyOn(queryClient, 'prefetchQuery').mockResolvedValue(undefined)

    // Classic shell: the global catalog shows its switches there, as on main.
    $shellChoice.set(false)

    try {
      await renderSkills()

      expect(await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })).toBeTruthy()
    } finally {
      cleanup()
      prefetchSpy.mockRestore()
      vi.unstubAllGlobals()
    }
  })

  it('routes agent tool enablement to its pinned remote owner', async () => {
    await renderSkills('researcher', 'homelab')

    // The switch names the action, so an enabled toolset offers to turn it off.
    const sw = await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })
    expect(sw.getAttribute('aria-checked')).toBe('true')

    await act(async () => {
      fireEvent.click(sw)
    })

    await waitFor(() => expect(setToolsetEnabled).toHaveBeenCalled())
    expect(setToolsetEnabled).toHaveBeenCalledWith('web', false, { connectionId: 'homelab', profile: 'researcher' })
  })

  it.each([undefined, 'researcher'])(
    'restores classic catalog switches and configuration without changing ownership (%s)',
    async profile => {
      await renderSkills(profile)
      act(() => $shellChoice.set(false))
      const toggle = await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })
      fireEvent.click(toggle)
      await waitFor(() => expect(setToolsetEnabled).toHaveBeenCalledWith('web', false, profile ?? 'default'))
      fireEvent.click(screen.getByRole('button', { name: 'Web Search' }))
      await waitFor(() => expect(getToolsetConfig).toHaveBeenCalledWith('web', profile ?? 'default'))
      act(() => $shellChoice.set(true))

      if (!profile) {
        expect(screen.queryByRole('switch')).toBeNull()
      }
    }
  )

  it('keeps the full off inventory reachable and writes only the pinned owner', async () => {
    getToolsets.mockResolvedValue(
      Array.from({ length: 10 }, (_, index) =>
        toolset({
          name: `optional_${index}`,
          label: `Optional ${index}`,
          enabled: false,
          available: false,
          configured: false
        })
      )
    )
    await renderSkills('researcher', 'homelab')
    await screen.findByRole('button', { name: 'Show all 10' })
    expect(screen.queryByRole('switch', { name: 'Turn Optional 9 toolset on' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show all 10' }))
    fireEvent.click(screen.getByRole('switch', { name: 'Turn Optional 9 toolset on' }))
    await waitFor(() =>
      expect(setToolsetEnabled).toHaveBeenCalledWith('optional_9', true, {
        connectionId: 'homelab',
        profile: 'researcher'
      })
    )
  })

  it.each([false, true])('scopes Tools per profile (V2=%s)', async enabled => {
    $shellChoice.set(enabled)
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

  it.each([false, true])('scopes Skills per profile (V2=%s)', async enabled => {
    $shellChoice.set(enabled)
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

    // The selector renders on the Skills tab too (Capabilities-wide); the catalog's sort lives in its Filters popover.
    const trigger = await screen.findByRole('combobox')
    await act(async () => {
      fireEvent.click(trigger)
    })
    const option = await screen.findByRole('option', { name: 'researcher' })
    await act(async () => {
      fireEvent.click(option)
    })

    // Skills refetch scoped to the picked profile...
    await waitFor(() => expect(getSkills).toHaveBeenCalledWith('researcher'))

    if (enabled) {
      expect(screen.queryByRole('switch', { name: 'web-research' })).toBeNull()
      expect(setSkillEnabled).not.toHaveBeenCalled()
    } else {
      fireEvent.click(screen.getByRole('switch', { name: 'web-research' }))
      await waitFor(() => expect(setSkillEnabled).toHaveBeenCalledWith('web-research', false, 'researcher'))
    }
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
    fireEvent.click(screen.getByRole('button', { name: 'Add gif-search' }))
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

    await renderSkills('researcher', 'homelab')

    fireEvent.click(await screen.findByRole('button', { name: 'Vision / Image Analysis: 1 tool' }))

    const link = await screen.findByRole('button', { name: /Choose vision model in Settings/ })
    await waitFor(() =>
      expect(getToolsetConfig).toHaveBeenCalledWith('vision', { connectionId: 'homelab', profile: 'researcher' })
    )

    await act(async () => {
      fireEvent.click(link)
    })

    // Internal route change into the Models section with the aux slot target —
    // consumed by ModelSettings' deep-link highlight. Never an external URL.
    await waitFor(() =>
      expect(navigateSpy).toHaveBeenCalledWith(
        '/settings?tab=config%3Amodel&aux=vision&agentProfile=researcher&panel=settings&connection=homelab'
      )
    )
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
          targetProfile: 'default',
          handle: 'inbox-bot-homelab'
        }
      ],
      sources: []
    })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { connections, getAgentRoster }

    try {
      await renderSkills()

      await waitFor(() => expect(getAgentRoster).toHaveBeenCalled())
      Element.prototype.scrollIntoView = vi.fn()
      const selector = await screen.findByRole('combobox')
      expect(within(selector).getByText('default — This device (current)')).toBeTruthy()
      fireEvent.click(selector)
      fireEvent.click(await screen.findByRole('option', { name: 'inbox-bot — Homelab' }))
      await waitFor(() => expect(getToolsets).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'default' }))
      // Connectors are managed on this page, so its tab offers no hand-off to the agent's settings.
      expect(screen.queryByRole('button', { name: TRANSLATIONS.en.skills.openAgentSettings })).toBeNull()
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('lists the built-in optional-skills catalog with Add buttons that route through the hub pipeline', async () => {
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

    // The row settles once installed skills load (the discovery hero can change
    // as they merge in), so act on the live, enabled switch.
    await screen.findByRole('button', { name: 'Add gif-search' })

    const install = await waitFor(() => {
      const row = screen.getByRole<HTMLButtonElement>('button', { name: 'Add gif-search' })

      expect(row.disabled).toBe(false)

      return row
    })

    expect(screen.queryByRole('switch', { name: 'web-research' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Add web-research' })).toBeNull()
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Added ascii-art' }).disabled).toBe(true)

    await act(async () => {
      fireEvent.click(install)
    })

    await waitFor(() =>
      expect(vi.mocked(installHubSkill)).toHaveBeenCalledWith('official/gifs/gif-search', expect.anything())
    )
  })

  it('drops community feed rows that only share a name with an installed skill', async () => {
    // Installs are name-keyed, so a community lookalike of an installed skill
    // can neither be added beside it nor is it the installed skill itself —
    // it must not surface as a second "installed" row under the same name.
    getSkills.mockResolvedValue([
      {
        name: 'docx',
        description: 'Bundled document tools',
        category: 'documents',
        enabled: true,
        usage: 0,
        provenance: 'bundled'
      }
    ])
    queryClient.setQueryData(
      ['public-catalog', 'skills'],
      parseCatalog('skills', [
        { name: 'docx', source: 'ClawHub', identifier: 'wordpro' },
        { name: 'gif-search', source: 'ClawHub', identifier: 'gif-raccoon' }
      ])
    )

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    // The feed settles first; then only the installed row keeps the name,
    // while the distinct community skill stays installable.
    await screen.findByRole('button', { name: 'Add gif-search' })
    expect(screen.getAllByRole('button', { name: 'docx' })).toHaveLength(1)
  })

  it('shows a ClawHub card as installed when its lock row is owner-qualified', async () => {
    // The lock records @owner/slug; the hub map also lists the row under the
    // card's clawhub/<slug> id, so the card takes the installed skill's switch.
    // my-pdf declares `name: pdf`, so its card cannot merge but still reads installed.
    const { installHubSkill } = await import('@/store/hub-actions')
    const record = (name: string) => ({ name, trust_level: 'community', scan_verdict: 'safe' })

    const installed = Object.fromEntries(
      [
        ['@tmstudio667-commits/agent-git-oracle', 'agent-git-oracle'],
        ['@liujiang817/my-pdf', 'my-pdf']
      ].flatMap(([id, name]) => [id, `clawhub/${name}`].map(key => [key, record(name)]))
    )

    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path === '/api/skills/hub/sources') {
        return { sources: [], index_available: false, featured: [], installed }
      }

      throw new Error(`unexpected ${path}`)
    })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    getSkills.mockResolvedValue(
      ['agent-git-oracle', 'pdf'].map(name => ({ name, description: name, enabled: true, usage: 0, provenance: 'hub' }))
    )
    queryClient.setQueryData(
      ['public-catalog', 'skills'],
      parseCatalog('skills', [
        { name: 'Agent Git Oracle', source: 'ClawHub', identifier: 'agent-git-oracle' },
        { name: 'My Pdf', source: 'ClawHub', identifier: 'my-pdf' }
      ])
    )

    try {
      await act(async () => {
        render(
          <QueryClientProvider client={queryClient}>
            <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
              <CapabilitiesView />
            </MemoryRouter>
          </QueryClientProvider>
        )
      })

      expect((await screen.findByRole<HTMLButtonElement>('button', { name: 'Added My Pdf' })).disabled).toBe(true)
      expect(screen.queryByRole('switch', { name: 'agent-git-oracle' })).toBeNull()
      expect(screen.getByRole('button', { name: 'Agent Git Oracle' })).toBeTruthy()
      expect(screen.queryByRole('button', { name: /^Add (Agent Git Oracle|My Pdf)$/ })).toBeNull()
      expect(installHubSkill).not.toHaveBeenCalled()
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })
})

interface PageAt {
  tab: 'connectors' | 'plugins' | 'skills' | 'toolsets'
  /** Agent settings: embedded, one capability per section. */
  embedded?: boolean
  fixedConnection?: string
  fixedProfile?: string
  owner?: { connectionId: string; profile: string }
}

async function renderAt({ tab, embedded, fixedConnection, fixedProfile, owner }: PageAt) {
  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[`/capabilities?tab=${tab}`]}>
          <SettingsOwnerContext.Provider value={owner}>
            <CapabilitiesView
              embedded={embedded}
              fixedConnection={fixedConnection}
              fixedMode={embedded ? tab : undefined}
              fixedProfile={fixedProfile}
            />
          </SettingsOwnerContext.Provider>
        </MemoryRouter>
      </QueryClientProvider>
    )
  })
}

const pluginLoads = (profile: string) => loadAgentPlugins.mock.calls.filter(([, loaded]) => loaded === profile)

describe('CapabilitiesView plugins and tab counts', { timeout: 60_000 }, () => {
  beforeEach(() => {
    $gatewayState.set('open')
  })

  it("classic lists a pinned remote agent's plugins as it always has, with no hand-off", async () => {
    $shellChoice.set(false)
    await renderAt({ tab: 'plugins', fixedConnection: 'homelab', fixedProfile: 'researcher' })

    await waitFor(() => expect(pluginLoads('researcher').length).toBeGreaterThan(0))
    expect(screen.queryByRole('button', { name: TRANSLATIONS.en.skills.openAgentSettings })).toBeNull()
  })

  it("an agent's own settings manage its plugins on its own backend; elsewhere the redesign hands off", async () => {
    const remote: PageAt = { fixedConnection: 'homelab', fixedProfile: 'researcher', tab: 'plugins' }
    await renderAt({ ...remote, embedded: true, owner: { connectionId: 'homelab', profile: 'researcher' } })

    await waitFor(() => expect(pluginLoads('researcher').length).toBeGreaterThan(0))
    expect(screen.queryByRole('button', { name: TRANSLATIONS.en.skills.openAgentSettings })).toBeNull()

    cleanup()
    loadAgentPlugins.mockClear()
    await renderAt({ ...remote, embedded: true })

    expect(await screen.findByRole('button', { name: TRANSLATIONS.en.skills.openAgentSettings })).toBeTruthy()
    expect(pluginLoads('researcher')).toHaveLength(0)
  })

  it('classic and embedded views never read the redesign tab counts', async () => {
    $shellChoice.set(false)
    await renderAt({ tab: 'toolsets' })
    await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })

    cleanup()
    $shellChoice.set(true)
    await renderAt({ tab: 'toolsets', embedded: true, fixedProfile: 'researcher' })
    await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })

    expect(loadAgentPlugins).not.toHaveBeenCalled()
    expect(getHermesConfigRecord).not.toHaveBeenCalled()

    // The redesigned page does count them.
    cleanup()
    await renderAt({ tab: 'toolsets' })
    await waitFor(() => expect(loadAgentPlugins).toHaveBeenCalled())
    expect(getHermesConfigRecord).toHaveBeenCalled()
  })

  it("the redesigned page does not count another machine's plugins through this window's gateway", async () => {
    await renderAt({ tab: 'toolsets', fixedConnection: 'homelab', fixedProfile: 'researcher' })
    await waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalled())

    expect(loadAgentPlugins).not.toHaveBeenCalled()
  })

  it("a toolset switched in an agent's list keeps its place, its focus and its row", async () => {
    const off = Array.from({ length: 9 }, (_, index) =>
      toolset({ name: `optional_${index}`, label: `Optional ${index}`, enabled: false })
    )

    const zeta = toolset({ name: 'zeta', label: 'Zeta' })
    getToolsets.mockResolvedValue([...off, zeta])
    setToolsetEnabled.mockImplementation(async () => {
      getToolsets.mockResolvedValue([...off, { ...zeta, enabled: false }])

      return { ok: true, name: 'zeta', enabled: false }
    })
    await renderAt({ tab: 'toolsets', fixedProfile: 'researcher' })

    const sw = await screen.findByRole('switch', { name: 'Turn Zeta toolset off' })
    const onSection = sw.closest('section')!
    sw.focus()
    fireEvent.click(sw)

    await waitFor(() => expect(getToolsets.mock.calls.length).toBeGreaterThan(1))
    const switched = await screen.findByRole('switch', { name: 'Turn Zeta toolset on' })
    // Same element, so focus stays put; still under On, so it never drops past the Off preview.
    expect(switched).toBe(sw)
    expect(sw.ownerDocument.activeElement).toBe(sw)
    expect(onSection.contains(switched)).toBe(true)
  })

  it.each([
    [false, 'inbox-bot'],
    [true, 'default']
  ])('picking a roster alias keeps its own row (redesign=%s) and reads %s', async (redesign, served) => {
    $shellChoice.set(redesign)
    Element.prototype.scrollIntoView = vi.fn()

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

    const agent = (connectionId: string, connectionLabel: string, profile: string) => ({
      connectionId,
      connectionKind: connectionId === 'local' ? 'local' : 'remote',
      connectionLabel,
      profile,
      targetProfile: 'default',
      handle: `${profile}-${connectionId}`
    })

    const getAgentRoster = vi.fn().mockResolvedValue({
      agents: [
        agent('local', 'This device', 'default'),
        agent('homelab', 'Homelab', 'default'),
        agent('homelab', 'Homelab', 'inbox-bot')
      ],
      sources: []
    })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { connections, getAgentRoster }

    try {
      await renderAt({ tab: 'toolsets' })
      fireEvent.click(await screen.findByRole('combobox'))
      fireEvent.click(await screen.findByRole('option', { name: 'inbox-bot — Homelab' }))

      await waitFor(() => expect(getToolsets).toHaveBeenCalledWith({ connectionId: 'homelab', profile: served }))
      expect(within(screen.getByRole('combobox')).getByText('inbox-bot — Homelab')).toBeTruthy()
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })
})
