import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { useI18n } from '@/i18n'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId } from '@/store/connections'
import type { SessionInfo } from '@/types/hermes'

import { gatewayLabel } from '../gateway-group-model'

import { withGatewayProjectRows } from './gateway-project-rows'
import type { SidebarProjectTree } from './workspace-groups'

/** The window gateway's project tree plus the rows other gateways own (`withGatewayProjectRows`). */
export function useGatewayProjectTree(tree: SidebarProjectTree[], live: SessionInfo[]): SidebarProjectTree[] {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const localLabel = t.settings.gateway.webappHostTitle

  return useMemo(() => {
    if (!registry) {
      return tree
    }

    // A bare row is the registry primary's, the owner routing resolves it to.
    const ownerOf = (connectionId: null | string | undefined) => connectionId?.trim() || registry.primary
    const treeOwner = ownerOf(activeConnectionId)

    return withGatewayProjectRows(
      tree,
      live,
      session => {
        const owner = ownerOf(session.connection_id)

        return owner && owner !== treeOwner ? owner : null
      },
      connectionId => gatewayLabel(connectionId, registry, localLabel)
    )
  }, [tree, live, registry, activeConnectionId, localLabel])
}
