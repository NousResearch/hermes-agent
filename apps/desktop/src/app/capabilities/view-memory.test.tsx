/**
 * Reopening Capabilities lands on the pane last used. The pane lives in the
 * route's `?tab=`, so without memory a reopen snaps back to Skills; with memory
 * it reopens on Plugins. An explicit `?tab=` still wins.
 */
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { resetViewMemory } from '@/store/view-memory'

vi.mock('./scope-selector', () => ({
  CapabilityScopeSelector: () => null,
  useCapabilityScope: () => ({ crossBackend: false, key: 'local', label: 'default', options: [], profile: null })
}))
vi.mock('./skills/skills-data', () => ({
  SKILLS_QUERY_KEY: ['skills'],
  skillSearchTerms: () => [],
  useSkillsQuery: () => ({ data: [], isError: false })
}))
vi.mock('./toolsets/toolsets-data', () => ({
  TOOLSETS_QUERY_KEY: ['toolsets'],
  toolsetSearchTerms: () => [],
  useToolsetsQuery: () => ({ data: [], isError: false }),
  visibleToolsetCount: () => 0
}))
vi.mock('./skills/skills-tab', () => ({ SkillsTab: () => <div data-testid="pane-skills" /> }))
vi.mock('./toolsets/toolsets-tab', () => ({ ToolsetsTab: () => <div data-testid="pane-toolsets" /> }))
vi.mock('./connectors/connectors-tab', () => ({ ConnectorsTab: () => <div data-testid="pane-connectors" /> }))
vi.mock('./plugins/plugins-tab', () => ({ PluginsTab: () => <div data-testid="pane-plugins" /> }))
vi.mock('./skills/embedded-hub-picker', () => ({ EmbeddedHubPicker: () => null }))
vi.mock('./toolsets/tool-calls', () => ({ refreshToolCalls: vi.fn() }))

const { CapabilitiesView } = await import('./index')

async function renderAt(pathname: string) {
  let result: ReturnType<typeof render>

  await act(async () => {
    result = render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[pathname]}>
          <CapabilitiesView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

const tab = (container: HTMLElement, id: string) => container.querySelector<HTMLElement>(`[data-tour="tab-${id}"]`)

beforeEach(resetViewMemory)

afterEach(() => {
  cleanup()
  queryClient.clear()
})

describe('Capabilities remembers its last pane', () => {
  it('reopens on the pane last used', async () => {
    const first = await renderAt('/capabilities')

    expect(tab(first.container, 'skills')!.getAttribute('data-active')).toBe('true')

    await act(async () => {
      fireEvent.click(tab(first.container, 'plugins')!)
    })
    expect(tab(first.container, 'plugins')!.getAttribute('data-active')).toBe('true')
    expect(screen.getByTestId('pane-plugins')).toBeTruthy()

    cleanup()

    // Reopen with no `?tab=` — memory decides.
    const second = await renderAt('/capabilities')
    expect(tab(second.container, 'plugins')!.getAttribute('data-active')).toBe('true')
    expect(screen.getByTestId('pane-plugins')).toBeTruthy()
  })

  it('lets an explicit ?tab= win over the remembered pane', async () => {
    const first = await renderAt('/capabilities')

    await act(async () => {
      fireEvent.click(tab(first.container, 'plugins')!)
    })

    cleanup()

    const second = await renderAt('/capabilities?tab=toolsets')
    expect(tab(second.container, 'toolsets')!.getAttribute('data-active')).toBe('true')
    expect(screen.getByTestId('pane-toolsets')).toBeTruthy()
  })
})
