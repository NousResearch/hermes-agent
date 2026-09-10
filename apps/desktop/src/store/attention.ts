import { computed } from 'nanostores'

import { type AttentionItem, type AttentionOwner, buildAttentionItems } from '@/app/command-center/attention'

import { $messagingSessions, $sessions, getSessionOwnerHints } from './session'
import { $sessionDotStateById } from './session-dot-state'
import { $attentionSessionIds } from './session-states'

/** One attention list for the Desktop titlebar and Command Center. */
export const $attentionItems = computed(
  [$attentionSessionIds, $sessionDotStateById, $sessions, $messagingSessions],
  (attentionSessionIds, dotStates, sessions, messagingSessions): AttentionItem[] => {
    const attentionOwners: Record<string, AttentionOwner[]> = {}

    for (const sessionId of attentionSessionIds) {
      const owners = getSessionOwnerHints(sessionId).map(route => ({
        connectionId: route.connectionId.trim() || undefined,
        profile: route.profile.trim() || 'default'
      }))

      if (owners.length > 0) {
        attentionOwners[sessionId] = owners
      }
    }

    return buildAttentionItems({
      attentionOwners,
      attentionSessionIds,
      dotStates,
      messagingSessions,
      sessions
    })
  }
)

export const $attentionCount = computed($attentionItems, items => items.length)
