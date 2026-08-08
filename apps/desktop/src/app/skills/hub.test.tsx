// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import type { McpCatalogEntry, SkillHubSourcesResponse } from '@/types/hermes'

const activeProfile = atom('default')
const activeSessionId = atom<string | null>('session-1')
const gateway = atom<null | { request: ReturnType<typeof vi.fn> }>(null)
const getMcpCatalog = vi.fn()
const getSkillHubSources = vi.fn()
const searchSkillsHub = vi.fn()

vi.mock('@/hermes', () => ({
  authMcpServer: vi.fn(),
  getActionStatus: vi.fn(),
  getMcpCatalog: () => getMcpCatalog(),
  getMcpOAuthFlow: vi.fn(),
  getSkillHubSources: () => getSkillHubSources(),
  installMcpCatalogEntry: vi.fn(),
  installSkillFromHub: vi.fn(),
  previewSkillHub: vi.fn(),
  scanSkillHub: vi.fn(),
  searchSkillsHub: (term: string, source: string) => searchSkillsHub(term, source),
  setMcpServerEnabled: vi.fn(),
  uninstallSkillFromHub: vi.fn(),
  updateSkillsFromHub: vi.fn()
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/mcp-dashboard-oauth', () => ({ completeMcpDesktopOAuth: vi.fn() }))
vi.mock('@/store/gateway', () => ({ $gateway: gateway }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: activeProfile,
  normalizeProfileKey: (profile: null | string | undefined) => profile?.trim() || 'default'
}))
vi.mock('@/store/session', () => ({ $activeSessionId: activeSessionId }))

function integration(patch: Partial<McpCatalogEntry> = {}): McpCatalogEntry {
  return {
    args: [],
    authenticated: null,
    auth_type: 'none',
    bootstrap: [],
    command: null,
    default_enabled: null,
    description: 'Work with Linear issues and projects.',
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
    url: 'https://mcp.linear.app',
    ...patch
  }
}

function sources(): SkillHubSourcesResponse {
  return {
    featured: [
      {
        description: 'Write clear, effective prompts for a task.',
        identifier: 'official/prompt-writer',
        name: 'Prompt writer',
        repo: null,
        source: 'official',
        tags: ['writing'],
        trust_level: 'builtin'
      }
    ],
    index_available: true,
    installed: {
      'official/local-helper': {
        name: 'Local helper',
        scan_verdict: null,
        trust_level: 'builtin'
      }
    },
    sources: [{ id: 'official', label: 'Official', searchable: true }]
  }
}

async function renderHub() {
  const { SkillsHub } = await import('./hub')
  let result: ReturnType<typeof render>

  await act(async () => {
    result = render(
      <QueryClientProvider client={queryClient}>
        <SkillsHub query="" />
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  activeProfile.set('default')
  activeSessionId.set('session-1')
  gateway.set(null)
  getMcpCatalog.mockResolvedValue({
    diagnostics: [],
    entries: [integration(), integration({ installed: true, name: 'notion' })]
  })
  getSkillHubSources.mockResolvedValue(sources())
  searchSkillsHub.mockResolvedValue({ installed: {}, results: [], source_counts: {}, timed_out: [] })
})

afterEach(async () => {
  const { $hubActions, $hubActiveLog, $hubInstalledOverride } = await import('@/store/hub-actions')

  $hubActions.set({})
  $hubActiveLog.set(null)
  $hubInstalledOverride.set({})
  cleanup()
  queryClient.clear()
  vi.clearAllMocks()
})

describe('SkillsHub', () => {
  it('combines installed skills and integrations in one readable discovery page', async () => {
    await renderHub()

    await screen.findByText('Featured skills')
    expect(screen.getByText('Local helper')).toBeTruthy()
    expect(screen.getByText('Notion')).toBeTruthy()
    expect(screen.getByText('Prompt writer')).toBeTruthy()
    expect(await screen.findByRole('button', { name: 'Install Linear' })).toBeTruthy()
  })

  it('filters the same Hub between skills and integrations', async () => {
    await renderHub()

    await screen.findByText('Prompt writer')
    fireEvent.click(screen.getByRole('button', { name: 'Integrations' }))

    await waitFor(() => expect(screen.queryByText('Prompt writer')).toBeNull())
    expect(await screen.findByRole('button', { name: 'Install Linear' })).toBeTruthy()
  })
})
