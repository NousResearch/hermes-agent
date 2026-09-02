import { atom } from 'nanostores'

import type { GatewayRequest } from '@/store/agent-plugins'

export interface MarketplacePlugin {
  compatible: boolean
  description: string
  display_name: string
  incompatibility_reason: string
  maintainer: string
  name: string
  source_id: string
  source_name: string
  version: string
}

export interface PluginMarketplace {
  id: string
  name: string
  url: string
  available: boolean
  stale: boolean
  entries: MarketplacePlugin[]
  error?: string
}

export const $pluginMarketplaces = atom<PluginMarketplace[]>([])
export const $pluginMarketplaceScope = atom<string | null>(null)
export const $pluginMarketplacesStatus = atom<'idle' | 'loading' | 'ready' | 'error'>('idle')
export const $pluginMarketplacesError = atom<string | null>(null)
export const $pluginMarketplaceBusy = atom<string | null>(null)

const scoped = (params: Record<string, unknown>, profile?: string | null) => (profile ? { ...params, profile } : params)

let loadGeneration = 0

export async function loadPluginMarketplaces(
  request: GatewayRequest,
  profile?: string | null,
  force = false,
  scopeKey = profile ?? 'default'
): Promise<void> {
  const generation = ++loadGeneration
  if ($pluginMarketplaceScope.get() !== scopeKey) {
    $pluginMarketplaceBusy.set(null)
    $pluginMarketplaces.set([])
    $pluginMarketplacesError.set(null)
    $pluginMarketplacesStatus.set('loading')
  }
  $pluginMarketplaceScope.set(scopeKey)
  if ($pluginMarketplacesStatus.get() !== 'ready') {
    $pluginMarketplacesStatus.set('loading')
  }

  try {
    const result = await request<{ marketplaces?: PluginMarketplace[] }>(
      'plugins.manage',
      scoped({ action: force ? 'marketplace_refresh' : 'marketplaces' }, profile)
    )
    if (generation !== loadGeneration || $pluginMarketplaceScope.get() !== scopeKey) {
      return
    }
    $pluginMarketplaces.set(result.marketplaces ?? [])
    $pluginMarketplacesError.set(null)
    $pluginMarketplacesStatus.set('ready')
  } catch (error) {
    if (generation !== loadGeneration || $pluginMarketplaceScope.get() !== scopeKey) {
      return
    }
    $pluginMarketplacesError.set(error instanceof Error ? error.message : String(error))
    $pluginMarketplacesStatus.set('error')
  }
}

export async function addPluginMarketplace(
  request: GatewayRequest,
  url: string,
  profile?: string | null,
  scopeKey = profile ?? 'default'
): Promise<boolean> {
  $pluginMarketplaceBusy.set('add')

  try {
    await request('plugins.manage', scoped({ action: 'marketplace_add', url }, profile))
    if ($pluginMarketplaceScope.get() === scopeKey) {
      await loadPluginMarketplaces(request, profile, false, scopeKey)
    }
    return true
  } catch (error) {
    if ($pluginMarketplaceScope.get() === scopeKey) {
      $pluginMarketplacesError.set(error instanceof Error ? error.message : String(error))
    }
    return false
  } finally {
    if ($pluginMarketplaceScope.get() === scopeKey) {
      $pluginMarketplaceBusy.set(null)
    }
  }
}

export async function removePluginMarketplace(
  request: GatewayRequest,
  sourceId: string,
  profile?: string | null,
  scopeKey = profile ?? 'default'
): Promise<boolean> {
  $pluginMarketplaceBusy.set(sourceId)

  try {
    await request('plugins.manage', scoped({ action: 'marketplace_remove', source_id: sourceId }, profile))
    if ($pluginMarketplaceScope.get() === scopeKey) {
      await loadPluginMarketplaces(request, profile, false, scopeKey)
    }
    return true
  } catch (error) {
    if ($pluginMarketplaceScope.get() === scopeKey) {
      $pluginMarketplacesError.set(error instanceof Error ? error.message : String(error))
    }
    return false
  } finally {
    if ($pluginMarketplaceScope.get() === scopeKey) {
      $pluginMarketplaceBusy.set(null)
    }
  }
}
