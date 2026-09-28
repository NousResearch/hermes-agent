import { atom } from 'nanostores'

import type { GatewayRequest } from './agent-plugins'

export interface MarketplacePlugin {
  name: string
  display_name: string
  description: string
  version: string
  maintainer: string
  compatible: boolean
  incompatibility_reason: string
}

export interface PluginMarketplace {
  id: string
  name: string
  available: boolean
  stale: boolean
  entries: MarketplacePlugin[]
  error?: string
}

export const $pluginMarketplaces = atom<PluginMarketplace[]>([])
export const $pluginMarketplaceScope = atom<string | null>(null)
export const $pluginMarketplacesError = atom<string | null>(null)
export const $pluginMarketplaceBusy = atom<string | null>(null)
let generation = 0
export const pluginMarketplaceGeneration = () => generation

const scoped = (params: Record<string, unknown>, profile: string | null) => (profile ? { ...params, profile } : params)
// ponytail: a single active snapshot; connection + profile and generation fence stale responses.
export const marketplaceScopeKey = (connectionId: string | null, profile: string | null) =>
  JSON.stringify([connectionId ?? 'local', profile ?? 'default'])

export async function loadPluginMarketplaces(request: GatewayRequest, profile: string | null, connectionId: string | null, refresh = false) {
  const scope = marketplaceScopeKey(connectionId, profile)
  const current = ++generation

  if ($pluginMarketplaceScope.get() !== scope) {
    $pluginMarketplaces.set([])
    $pluginMarketplacesError.set(null)
    $pluginMarketplaceBusy.set(null)
  }

  $pluginMarketplaceScope.set(scope)
  if (refresh) $pluginMarketplaceBusy.set('refresh')

  try {
    const result = await request<{ marketplaces?: PluginMarketplace[] }>(
      'plugins.manage',
      scoped({ action: refresh ? 'marketplace_refresh' : 'marketplaces' }, profile)
    )

    if (current === generation) {
      $pluginMarketplaces.set(result.marketplaces ?? [])
      $pluginMarketplacesError.set(null)
    }
  } catch (error) {
    if (current === generation) {
      $pluginMarketplacesError.set(error instanceof Error ? error.message : String(error))
    }
  } finally {
    if (refresh && current === generation && $pluginMarketplaceBusy.get() === 'refresh') {
      $pluginMarketplaceBusy.set(null)
    }
  }
}

export async function addPluginMarketplace(request: GatewayRequest, url: string, profile: string | null, connectionId: string | null) {
  const scope = marketplaceScopeKey(connectionId, profile)
  const started = generation
  $pluginMarketplaceBusy.set('add')

  try {
    const result = await request<{ ok?: boolean; error?: string }>(
      'plugins.manage',
      scoped({ action: 'marketplace_add', url }, profile)
    )

    if (!result?.ok) {
      throw new Error(result?.error || 'Could not add marketplace')
    }

    if ($pluginMarketplaceScope.get() === scope && started === generation) {
      await loadPluginMarketplaces(request, profile, connectionId)
    }

    return true
  } catch (error) {
    if ($pluginMarketplaceScope.get() === scope && started === generation) {
      $pluginMarketplacesError.set(error instanceof Error ? error.message : String(error))
    }

    return false
  } finally {
    if ($pluginMarketplaceScope.get() === scope && (started === generation || started + 1 === generation)) {
      $pluginMarketplaceBusy.set(null)
    }
  }
}

export async function removePluginMarketplace(request: GatewayRequest, sourceId: string, profile: string | null, connectionId: string | null) {
  const scope = marketplaceScopeKey(connectionId, profile)
  const started = generation
  $pluginMarketplaceBusy.set(sourceId)
  try {
    const result = await request<{ ok?: boolean; removed?: boolean }>(
      'plugins.manage', scoped({ action: 'marketplace_remove', source_id: sourceId }, profile)
    )
    if (!result?.ok || !result.removed) throw new Error('Could not remove marketplace')
    if ($pluginMarketplaceScope.get() === scope && started === generation) {
      await loadPluginMarketplaces(request, profile, connectionId)
    }
    return true
  } catch (error) {
    if ($pluginMarketplaceScope.get() === scope && started === generation) {
      $pluginMarketplacesError.set(error instanceof Error ? error.message : String(error))
    }
    return false
  } finally {
    if ($pluginMarketplaceScope.get() === scope && (started === generation || started + 1 === generation)) {
      $pluginMarketplaceBusy.set(null)
    }
  }
}
