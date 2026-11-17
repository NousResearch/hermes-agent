// A desktop-only catalog entry (Accent Picker) installed as a standalone desktop plugin: no agent row
// carries its catalog_name, so the card matches it through the entry the installer recorded on disk
// (`packageOrigin.catalogName`, read from `.hermes-origin.json`). Before that it kept its Install "+".
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'

import { PageSearchShell } from '../../page-search-shell'
import { parseCatalog } from '../catalog/catalog-data'
import { $catalogCardView } from '../catalog/store'

import { PluginsTab } from './plugins-tab'

const requestGateway = vi.fn(async (): Promise<unknown> => ({ plugins: $agentPlugins.get() }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({ useGatewayRequest: () => ({ requestGateway }) }))

const ACCENT = {
  category: 'desktop',
  description: 'OKLCH accent colour picker',
  name: 'accent-picker',
  repo: 'https://github.com/NousResearch/hermes-official-plugins',
  sha: 'a'.repeat(40),
  subdir: 'accent-picker',
  tier: 'official'
}

beforeEach(() => {
  $gatewayState.set('open')
  $agentPlugins.set([])
  $agentPluginsStatus.set('ready')
  $catalogCardView.set(true)
  queryClient.setQueryData(['public-catalog', 'plugins'], parseCatalog('plugins', [ACCENT]))
})

afterEach(() => {
  cleanup()
  $pluginRecords.set({})
  queryClient.clear()
})

describe('PluginsTab catalog card for a standalone desktop plugin', () => {
  it('shows the installed plugin on its card, not as Install', async () => {
    $pluginRecords.set({
      accent: {
        id: 'accent',
        kind: 'disk',
        name: 'Accent Picker',
        packageOrigin: { catalogName: ACCENT.name, repo: ACCENT.repo, sha: ACCENT.sha },
        status: 'disabled'
      }
    })

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter>
            <PageSearchShell
              onSearchChange={() => undefined}
              searchHidden
              searchPlaceholder="Search plugins"
              searchValue=""
            >
              <PluginsTab onQueryChange={() => undefined} profile={null} query="" />
            </PageSearchShell>
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    expect(screen.queryByRole('button', { name: `Add ${ACCENT.name}` })).toBeNull()

    const card = screen
      .getAllByRole('article')
      .find(article => within(article).queryByRole('button', { name: ACCENT.name }))!

    expect(within(card).getByRole('switch', { name: 'Accent Picker' }).getAttribute('aria-checked')).toBe('false')
  })
})
