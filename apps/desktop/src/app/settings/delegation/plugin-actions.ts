import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { lookupPluginCatalogEntry } from '@/lib/plugin-catalog'
import type { GatewayRequest } from '@/store/agent-plugins'
import {
  $agentPlugins,
  $agentPluginsErrorOwner,
  $agentPluginsOwner,
  $agentPluginsStatus,
  loadAgentPlugins,
  toggleAgentPlugin
} from '@/store/agent-plugins'
import { notify } from '@/store/notifications'
import { openCatalogPluginInstall } from '@/store/plugin-catalog-install'

import { ADVANCED_DELEGATION_PLUGIN } from './routing-data'

/** Where the route backends' plugin stands for the Settings owner: drives the hint under a plugin mode. */
export type PluginState = 'disabled' | 'loading' | 'missing' | 'not_loaded'

/** The Advanced Delegation plugin's state for `owner` (the Settings "Applies to" profile; null = launch profile). */
export function useAdvancedDelegationPlugin(
  request: GatewayRequest,
  owner: null | string,
  ownerIdentity: string | undefined
) {
  const rows = useStore($agentPlugins)
  const loadedFor = useStore($agentPluginsOwner)
  const errorFor = useStore($agentPluginsErrorOwner)
  const status = useStore($agentPluginsStatus)
  const owned = loadedFor === ownerIdentity

  useEffect(() => {
    if (ownerIdentity !== undefined && !owned) {
      void loadAgentPlugins(request, owner, ownerIdentity)
    }
  }, [owned, owner, ownerIdentity, request])

  const row = owned
    ? rows.find(r => r.name === ADVANCED_DELEGATION_PLUGIN || r.key === ADVANCED_DELEGATION_PLUGIN)
    : undefined

  const state: PluginState = !owned
    ? status === 'error' && errorFor === ownerIdentity
      ? 'missing'
      : 'loading'
    : !row
      ? 'missing'
      : row.status === 'enabled'
        ? 'not_loaded'
        : 'disabled'

  return { row, state }
}

/** Open the reviewed catalog install dialog for the owner; a catalog without the entry says so. */
export async function installAdvancedDelegation(owner: null | string, unavailableMessage: string) {
  const result = await lookupPluginCatalogEntry(ADVANCED_DELEGATION_PLUGIN)

  if (!result.ok) {
    notify({ kind: 'error', message: unavailableMessage })

    return
  }

  openCatalogPluginInstall(result.entry, owner)
}

export function enableAdvancedDelegation(
  request: GatewayRequest,
  key: string,
  owner: null | string,
  ownerIdentity: string | undefined,
  failMessage: string
) {
  return toggleAgentPlugin(request, key, true, failMessage, owner, ownerIdentity)
}
