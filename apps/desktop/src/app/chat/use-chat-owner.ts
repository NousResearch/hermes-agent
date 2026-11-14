import { useStore } from '@nanostores/react'

import { tileOwnerRoute } from '@/app/chat/session-tile-owner'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $activeGatewayProfile, $newChatRoute } from '@/store/profile'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $unlistedSessionOwnerRows,
  knownSessionOwner,
  sessionMatchesStoredId
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

export interface ChatOwner {
  /** The chat's list row, when a list has it. */
  row: SessionInfo | undefined
  /** The agent (profile) that owns the chat; undefined while an existing chat's owner is unknown. */
  profile: string | undefined
  /** The connection that owns it (undefined: the ambient source). */
  connectionId: string | undefined
}

/** Who owns a chat (or the draft when `sessionId` is null): the tile's own route, else the owner the
 *  lists know, else for a draft the agent pinned for it. An unresolved existing chat never borrows
 *  the ambient agent. */
export function useChatOwner(sessionId: null | string): ChatOwner {
  const tiles = useStore($sessionTiles)
  const profile = useStore($activeGatewayProfile)
  const primary = useStore($primaryGatewayRoute)
  // A draft belongs to the agent pinned for it (session.create reads the same route), which the
  // ambient profile only follows once that agent's backend is in front.
  const draft = useStore($newChatRoute)

  const rows = [
    ...useStore($sessions),
    ...useStore($cronSessions),
    ...useStore($messagingSessions),
    ...useStore($unlistedSessionOwnerRows)
  ]

  const row = sessionId ? rows.find(session => sessionMatchesStoredId(session, sessionId)) : undefined
  const route = sessionId ? (tileOwnerRoute(tiles, rows, sessionId) ?? knownSessionOwner(rows, sessionId)) : undefined

  const ownerProfile =
    typeof route === 'string' ? route : (route?.profile ?? (!sessionId ? (draft?.profile ?? profile) : undefined))

  const connectionId =
    route && typeof route === 'object'
      ? route.connectionId
      : sessionId
        ? (primary.connectionId ?? undefined)
        : (draft?.connectionId ?? undefined)

  return { row, profile: ownerProfile, connectionId }
}
