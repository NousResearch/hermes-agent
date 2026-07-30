import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useCallback } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { toggleMessageReaction } from '@/store/reactions'
import { $reactionsEnabled } from '@/store/reactions-enabled'
import { $agentReactions, $localReactions, mergeReactions, setLocalReaction } from '@/store/reactions-local'
import type { MessageReaction } from '@/types/hermes'

// Stable empty identity — a fresh [] per render would re-run every consumer.
const EMPTY_REACTIONS: MessageReaction[] = []

/** Paint the tapback locally, then persist behind it. */
function commitReaction(
  messageId: string,
  role: ChatMessage['role'],
  rowId: number | undefined,
  reactions: MessageReaction[],
  emoji: null | string
): void {
  // Flip the UI immediately — a tapback is direct manipulation and must never
  // wait on a round-trip. Persistence follows in the background.
  setLocalReaction(messageId, emoji)
  void toggleMessageReaction({ id: messageId, role, rowId, reactions } as ChatMessage, emoji)
}

/**
 * A message's reactions and the one way to change them.
 *
 * Reads the durable list off `metadata.custom`, layers this window's live
 * overlays on top (the user's own click, the agent's mid-turn event), and
 * hands back a `react` that paints locally first and persists behind it.
 * Shared by the assistant footer slot and the user bubble's picker so both
 * apply identical tapback semantics.
 */
export function useMessageReactions(
  messageId: string,
  role: ChatMessage['role']
): {
  enabled: boolean
  react: (emoji: null | string) => void
  reactions: MessageReaction[]
} {
  const reactions = useAuiState(s => {
    const custom = (s.message.metadata?.custom ?? {}) as { reactions?: MessageReaction[] }

    return custom.reactions ?? EMPTY_REACTIONS
  })

  const rowId = useAuiState(s => {
    const custom = (s.message.metadata?.custom ?? {}) as { rowId?: number }

    return custom.rowId
  })

  const enabled = useStore($reactionsEnabled)
  const localAll = useStore($localReactions)
  const agentLive = useStore($agentReactions)

  return {
    enabled,
    react: useCallback(
      (emoji: null | string) => commitReaction(messageId, role, rowId, reactions, emoji),
      [messageId, reactions, role, rowId]
    ),
    reactions: mergeReactions(reactions, localAll[messageId], rowId === undefined ? undefined : agentLive[rowId])
  }
}
