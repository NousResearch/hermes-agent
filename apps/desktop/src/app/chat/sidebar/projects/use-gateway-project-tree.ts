import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { useI18n } from '@/i18n'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId } from '@/store/connections'
import type { SessionInfo } from '@/types/hermes'

import { gatewayLabel } from '../gateway-group-model'

import { withGatewayProjectRows } from './gateway-project-rows'
import type { SessionOwnerKey, SidebarProjectTree } from './workspace-groups'

export interface SessionOwners {
  /** The gateway owning a row when it is not the window's (whose tree files its own rows). */
  foreignOwner: (session: SessionInfo) => null | string
  /** A row's owning (gateway, profile): rows of two owners never merge under one session id. */
  ownerKey: SessionOwnerKey
}

const profileOf = (session: SessionInfo): string => (session.profile || '').trim() || 'default'

/** Who owns each sidebar row, resolved the way session routing resolves it. */
export function useSessionOwners(): SessionOwners {
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)

  return useMemo(() => {
    // A bare row is the registry primary's, the owner routing resolves it to.
    const ownerOf = (connectionId: null | string | undefined) => connectionId?.trim() || registry?.primary || ''
    const treeOwner = ownerOf(activeConnectionId)

    return {
      foreignOwner: session => {
        const owner = registry ? ownerOf(session.connection_id) : ''

        return owner && owner !== treeOwner ? owner : null
      },
      ownerKey: session => JSON.stringify([ownerOf(session.connection_id), profileOf(session)])
    }
  }, [registry, activeConnectionId])
}

/** The window gateway's project tree plus the rows other gateways own (`withGatewayProjectRows`),
 *  and the row ownership it was filed by. */
export function useGatewayProjectTree(
  tree: SidebarProjectTree[],
  live: SessionInfo[]
): { owners: SessionOwners; ownerKey: SessionOwnerKey; tree: SidebarProjectTree[] } {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const owners = useSessionOwners()
  const localLabel = t.settings.gateway.webappHostTitle

  const merged = useMemo(() => {
    if (!registry) {
      return tree
    }

    return withGatewayProjectRows(tree, live, owners.foreignOwner, connectionId =>
      gatewayLabel(connectionId, registry, localLabel)
    )
  }, [tree, live, registry, owners, localLabel])

  return { owners, ownerKey: owners.ownerKey, tree: merged }
}
