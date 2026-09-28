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

const scoped = (params: Record<string, unknown>, profile: string | null) => (profile ? { ...params, profile } : params)

export async function loadPluginMarketplaces(request: GatewayRequest, profile: string | null, refresh = false) {
  const scope = profile ?? 'default'
  const current = ++generation

  if ($pluginMarketplaceScope.get() !== scope) {
    $pluginMarketplaces.set([])
    $pluginMarketplacesError.set(null)
    $pluginMarketplaceBusy.set(null)
  }

  $pluginMarketplaceScope.set(scope)

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
  }
}

export async function addPluginMarketplace(request: GatewayRequest, url: string, profile: string | null) {
  const scope = profile ?? 'default'
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
      await loadPluginMarketplaces(request, profile)
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
