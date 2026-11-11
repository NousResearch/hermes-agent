import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import {
  $agentPlugins,
  $agentPluginsOwnerKey,
  $agentPluginsProfile,
  $agentPluginsStatus,
  type AgentPluginRow,
  type GatewayRequest,
  isCurrentAgentPluginRequest,
  loadAgentPlugins,
  toggleAgentPlugin
} from '@/store/agent-plugins'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'

import { useSettingsSearchCatalog } from './use-settings-search'

const requestGateway = vi.fn()

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<object>()),
  getEnvVars: async () => ({}),
  getHermesConfigSchema: async () => ({ fields: {} })
}))

vi.mock('../hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: {} })
}))

const row = (key: string, status: AgentPluginRow['status'] = 'enabled'): AgentPluginRow => ({
  key,
  name: key,
  description: '',
  source: 'user',
  status,
  version: '1'
})

let client: QueryClient

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  requestGateway.mockReset()
  $gatewayState.set('open')
  $activeGatewayProfile.set('default')
  $pluginRecords.set({})
  $agentPlugins.set([])
  $agentPluginsOwnerKey.set(undefined)
  $agentPluginsProfile.set(undefined)
  $agentPluginsStatus.set('idle')
})

afterEach(() => {
  cleanup()
  client.clear()
  $activeGatewayProfile.set('default')
  $agentPlugins.set([])
  $agentPluginsOwnerKey.set(undefined)
  $agentPluginsProfile.set(undefined)
  $agentPluginsStatus.set('idle')
})

describe('command palette plugin inventory', () => {
  it.each([undefined, 'local/research'])(
    'leaves the editor inventory and rescan authority intact (owner: %s)',
    async ownerKey => {
      const editor = Object.assign(vi.fn().mockResolvedValue({ plugins: [row('editor-plugin')] }), { ownerKey })
      const profile = ownerKey ? 'research' : null

      await loadAgentPlugins(editor as GatewayRequest, profile)
      const inventory = $agentPlugins.get()
      requestGateway.mockResolvedValue({ plugins: [row('search-plugin')] })

      const search = renderHook(() => useSettingsSearchCatalog(true), { wrapper: Wrapper })

      await waitFor(() =>
        expect(search.result.current.pluginEntries.map(entry => entry.plugin)).toEqual(['search-plugin'])
      )
      expect($agentPlugins.get()).toBe(inventory)
      expect($agentPluginsOwnerKey.get()).toBe(ownerKey)
      expect($agentPluginsProfile.get()).toBe(profile)
      expect($agentPluginsStatus.get()).toBe('ready')
      expect(isCurrentAgentPluginRequest(editor as GatewayRequest)).toBe(true)
      expect(editor).toHaveBeenCalledTimes(1)
      expect(requestGateway).toHaveBeenCalledTimes(1)

      search.unmount()

      // Rescan uses this same ownership guard before reloading the editor.
      if (isCurrentAgentPluginRequest(editor as GatewayRequest)) {
        await loadAgentPlugins(editor as GatewayRequest, profile)
      }

      expect(editor).toHaveBeenCalledTimes(2)
      expect($agentPlugins.get()).toEqual(inventory)
    }
  )

  it('keeps a late palette read and an old-owner toggle from replacing a newly pinned inventory', async () => {
    const toggle = Promise.withResolvers<{ ok: boolean; plugin: AgentPluginRow }>()
    const lookup = Promise.withResolvers<{ plugins: AgentPluginRow[] }>()

    const first = Object.assign(
      vi
        .fn()
        .mockResolvedValueOnce({ plugins: [row('demo')] })
        .mockReturnValueOnce(toggle.promise),
      { ownerKey: 'first/research' }
    )

    const second = Object.assign(vi.fn().mockResolvedValue({ plugins: [row('demo')] }), {
      ownerKey: 'second/research'
    })

    await loadAgentPlugins(first as GatewayRequest, 'research')
    const pendingToggle = toggleAgentPlugin(first as GatewayRequest, 'demo', false, 'Failed', 'research')
    requestGateway.mockReturnValueOnce(lookup.promise)
    const search = renderHook(() => useSettingsSearchCatalog(true), { wrapper: Wrapper })

    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))
    await act(async () => {
      await loadAgentPlugins(second as GatewayRequest, 'research')
      toggle.resolve({ ok: true, plugin: row('demo', 'disabled') })
      expect(await pendingToggle).toBe(true)
      lookup.resolve({ plugins: [row('search-plugin')] })
    })

    await waitFor(() =>
      expect(search.result.current.pluginEntries.map(entry => entry.plugin)).toEqual(['search-plugin'])
    )
    expect($agentPlugins.get()).toEqual([expect.objectContaining({ key: 'demo', status: 'enabled' })])
    expect($agentPluginsOwnerKey.get()).toBe('second/research')
    expect(isCurrentAgentPluginRequest(second as GatewayRequest)).toBe(true)
    expect(first).toHaveBeenCalledTimes(2)
    expect(second).toHaveBeenCalledTimes(1)
  })

  it('does not publish a late search response into a different active profile', async () => {
    const first = Promise.withResolvers<{ plugins: AgentPluginRow[] }>()
    requestGateway.mockReturnValueOnce(first.promise).mockResolvedValueOnce({ plugins: [row('research-plugin')] })
    const search = renderHook(() => useSettingsSearchCatalog(true), { wrapper: Wrapper })

    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))
    act(() => $activeGatewayProfile.set('research'))
    await waitFor(() =>
      expect(search.result.current.pluginEntries.map(entry => entry.plugin)).toEqual(['research-plugin'])
    )

    await act(async () => first.resolve({ plugins: [row('default-plugin')] }))

    expect(search.result.current.pluginEntries.map(entry => entry.plugin)).toEqual(['research-plugin'])
    expect($agentPluginsProfile.get()).toBeUndefined()
    expect($agentPlugins.get()).toEqual([])
  })
})
