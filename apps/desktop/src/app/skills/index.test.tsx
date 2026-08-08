// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import type * as ReactRouterDom from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'

const getSkills = vi.fn()
const getToolsets = vi.fn()
const setSkillEnabled = vi.fn()
const setToolsetEnabled = vi.fn()
const getToolsetConfig = vi.fn()
const selectToolsetProvider = vi.fn()
const getUsageAnalytics = vi.fn()
const getMcpCatalog = vi.fn()

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
  getMcpCatalog: () => getMcpCatalog()
}))

// Notifications hit nanostores/timers we don't care about here.
vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
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

async function renderSkills(initialRoute = '/skills?tab=installed&installed=tools') {
  const { SkillsView } = await import('./index')
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      // SkillsView reads skills/toolsets via useQuery, so it needs a provider.
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[initialRoute]}>
          <SkillsView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  getSkills.mockResolvedValue([])
  getToolsets.mockResolvedValue([toolset()])
  setToolsetEnabled.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: true, active_provider: null, providers: [] })
  getUsageAnalytics.mockResolvedValue({ tools: [] })
  getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  // Shared singleton client — drop cached skills/toolsets so each test refetches.
  queryClient.clear()
})

describe('SkillsView toolset management', () => {
  it('redirects a retired MCP deep link to Installed MCP', async () => {
    await renderSkills('/skills?tab=mcp&server=linear')

    await waitFor(() =>
      expect(navigateSpy).toHaveBeenCalledWith('/skills?tab=installed&installed=mcp&server=linear', { replace: true })
    )
  })

  it('shows installed MCP servers as a third Installed category', async () => {
    getMcpCatalog.mockResolvedValue({
      diagnostics: [],
      entries: [
        {
          args: [],
          authenticated: null,
          auth_type: 'none',
          bootstrap: [],
          command: null,
          default_enabled: null,
          description: 'Work with Notion pages.',
          enabled: true,
          install_ref: null,
          install_url: null,
          installed: true,
          name: 'notion',
          needs_install: false,
          post_install: '',
          required_env: [],
          source: 'https://notion.so',
          transport: 'http',
          url: 'https://mcp.notion.com'
        },
        {
          args: [],
          authenticated: null,
          auth_type: 'none',
          bootstrap: [],
          command: null,
          default_enabled: null,
          description: 'Work with Linear issues.',
          enabled: false,
          install_ref: null,
          install_url: null,
          installed: false,
          name: 'linear',
          needs_install: false,
          post_install: '',
          required_env: [],
          source: 'https://linear.app',
          transport: 'http',
          url: 'https://mcp.linear.app'
        }
      ]
    })

    await renderSkills('/skills?tab=installed&installed=mcp')

    expect(await screen.findByRole('button', { name: 'Ready Notion' })).toBeTruthy()
    expect(screen.queryByText('Linear')).toBeNull()
  })

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
