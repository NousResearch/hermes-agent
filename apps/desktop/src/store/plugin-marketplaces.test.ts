import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $agentPluginBusy,
  $agentPluginScope,
  $agentPlugins,
  loadAgentPlugins,
  updateAgentPlugin,
  type GatewayRequest
} from '@/store/agent-plugins'
import {
  $pluginMarketplaces,
  $pluginMarketplaceScope,
  $pluginMarketplacesStatus,
  loadPluginMarketplaces,
  type PluginMarketplace
} from '@/store/plugin-marketplaces'

const source = (name: string): PluginMarketplace => ({
  available: true,
  entries: [],
  id: name,
  name,
  stale: false,
  url: `https://github.com/example/${name}`
})

describe('plugin marketplace store', () => {
  beforeEach(() => {
    $pluginMarketplaces.set([])
    $pluginMarketplaceScope.set(null)
    $pluginMarketplacesStatus.set('idle')
  })

  it('ignores a slow response from the previous profile', async () => {
    const resolvers = new Map<string, (value: unknown) => void>()
    const request = vi.fn(
      async (_method: string, params?: Record<string, unknown>) =>
        await new Promise(resolve => resolvers.set(String(params?.profile), resolve))
    ) as unknown as GatewayRequest

    const firstOldLoad = loadPluginMarketplaces(request, 'old-profile')
    resolvers.get('old-profile')?.({ marketplaces: [source('old')] })
    await firstOldLoad
    expect($pluginMarketplaces.get().map(item => item.name)).toEqual(['old'])

    const staleOldLoad = loadPluginMarketplaces(request, 'old-profile')
    const newLoad = loadPluginMarketplaces(request, 'new-profile')
    expect($pluginMarketplaces.get()).toEqual([])
    expect($pluginMarketplacesStatus.get()).toBe('loading')
    resolvers.get('new-profile')?.({ marketplaces: [source('new')] })
    await newLoad
    resolvers.get('old-profile')?.({ marketplaces: [source('stale-old')] })
    await staleOldLoad

    expect($pluginMarketplaces.get().map(item => item.name)).toEqual(['new'])
  })

  it('isolates same-named profiles on different connections', async () => {
    let resolveOld: ((value: unknown) => void) | undefined
    const oldRequest = vi.fn(
      async () => await new Promise(resolve => (resolveOld = resolve))
    ) as unknown as GatewayRequest
    const newRequest = vi.fn(async () => ({ marketplaces: [source('new')] })) as unknown as GatewayRequest

    const oldLoad = loadPluginMarketplaces(oldRequest, 'same', false, 'old::same')
    await loadPluginMarketplaces(newRequest, 'same', false, 'new::same')
    resolveOld?.({ marketplaces: [source('old')] })
    await oldLoad

    expect($pluginMarketplaceScope.get()).toBe('new::same')
    expect($pluginMarketplaces.get().map(item => item.name)).toEqual(['new'])
  })

  it('does not publish an update continuation after the selected backend changes', async () => {
    let resolveUpdate: ((value: unknown) => void) | undefined
    const request = vi.fn(
      async () => await new Promise(resolve => (resolveUpdate = resolve))
    ) as unknown as GatewayRequest
    const current = {
      description: '',
      key: 'demo',
      name: 'demo-b',
      source: 'git',
      status: 'enabled' as const,
      version: '2.0.0'
    }
    $agentPluginScope.set('connection-a::same')

    const update = updateAgentPlugin(request, 'demo', 'failed', 'same', 'connection-a::same')
    $agentPluginScope.set('connection-b::same')
    $agentPlugins.set([current])
    resolveUpdate?.({ ok: true, unchanged: false })

    expect(await update).toBe(false)
    expect($agentPlugins.get()).toEqual([current])
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('does not report update success when scope changes during refresh', async () => {
    let resolveRefresh: ((value: unknown) => void) | undefined
    let calls = 0
    const request = vi.fn(async () => {
      calls += 1
      if (calls === 1) {
        return { ok: true, unchanged: false }
      }
      return await new Promise(resolve => (resolveRefresh = resolve))
    }) as unknown as GatewayRequest
    $agentPluginScope.set('connection-a::same')

    const update = updateAgentPlugin(request, 'demo', 'failed', 'same', 'connection-a::same')
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    $agentPluginScope.set('connection-b::same')
    resolveRefresh?.({ plugins: [] })

    expect(await update).toBe(false)
    expect($agentPluginScope.get()).toBe('connection-b::same')
  })

  it('clears plugin busy state when the selected backend changes', async () => {
    $agentPluginScope.set('old::same')
    $agentPluginBusy.set('demo')
    const request = vi.fn(async () => ({ plugins: [] })) as unknown as GatewayRequest

    await loadAgentPlugins(request, 'same', 'new::same')

    expect($agentPluginBusy.get()).toBeNull()
  })
})
