import { computed } from 'nanostores'

import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'
import { sessionApprovalRequests } from '@/store/prompts'

import { $backworkspaceWaiting } from './ask'
import { backworkspaceOwnerKey } from './page'

/**
 * The session this window's page is waiting on, while a question of its own is
 * in flight.
 *
 * Keyed by the page rather than by the agent, because the question may have
 * gone to a bot on another profile: what the reader in front of this page has
 * to answer is whatever the agent THEY asked is waiting for.
 */
export const $backworkspaceSessionId = computed(
  [$backworkspaceWaiting, $activeConnectionId, $activeGatewayProfile],
  (waiting, connectionId, profile) => waiting[backworkspaceOwnerKey({ connectionId, profile })] ?? null
)

/** The approvals waiting on the page for `sessionId`, in the order they arrived. */
export function backworkspaceApprovalQueue(sessionId: null | string) {
  return sessionApprovalRequests(sessionId)
}
