import { useAui, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type MouseEvent, useCallback } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { triggerHaptic } from '@/lib/haptics'
import { QUICK_REACTIONS, toggleMessageReaction } from '@/store/reactions'
import { $reactionsEnabled } from '@/store/reactions-enabled'
import { $agentReactions, $localReactions, mergeReactions, setLocalReaction } from '@/store/reactions-local'
import type { MessageReaction } from '@/types/hermes'

// Stable empty identity — a fresh [] per render would re-run every consumer.
const EMPTY_REACTIONS: MessageReaction[] = []

/** The tapback a double-click lands: Apple's first Tapback, and ours. */
export const DOUBLE_CLICK_REACTION = QUICK_REACTIONS[0]

// Double-click means something else on these: links and controls act, inputs
// and code blocks select. The gesture only claims plain message body.
const NOT_A_TAPBACK = 'a, button, input, pre, select, textarea, [contenteditable="true"], [role="button"]'

/**
 * Is this double-click the "heart it" gesture?
 *
 * `detail === 2` keeps a triple-click (select-the-paragraph) from re-firing,
 * and anything the browser already gives a double-click meaning keeps it.
 */
export function isTapbackDoubleClick(event: { detail: number; target: EventTarget | null }): boolean {
  if (event.detail !== 2) {
    return false
  }

  const target = event.target

  return target instanceof Element ? !target.closest(NOT_A_TAPBACK) : true
}

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
 * Shared by the assistant footer slot, the user bubble's picker, and the
 * double-click gesture so all three apply identical tapback semantics.
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

/**
 * Double-click a message to heart it — the iMessage gesture.
 *
 * Reads the message's reaction state lazily at event time (the same trick the
 * footer uses for its text): the gesture renders nothing, so subscribing the
 * perf-sensitive message root to every reaction change would be pure cost.
 * Returns `undefined` while reactions are off, so the element carries no
 * listener at all.
 */
export function useTapbackDoubleClick(
  messageId: string,
  role: ChatMessage['role']
): ((event: MouseEvent<HTMLElement>) => void) | undefined {
  const enabled = useStore($reactionsEnabled)
  const messageRuntime = useAui().message

  const onDoubleClick = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (!isTapbackDoubleClick(event)) {
        return
      }

      // Double-click has already selected the word underneath — the tapback,
      // not a stray selection, is what the gesture meant.
      window.getSelection()?.removeAllRanges()
      triggerHaptic('selection')

      const custom = (messageRuntime.getState().metadata?.custom ?? {}) as {
        reactions?: MessageReaction[]
        rowId?: number
      }

      const reactions = custom.reactions ?? EMPTY_REACTIONS

      // Same toggle semantics as the picker: a second double-click retracts.
      const mine = mergeReactions(reactions, $localReactions.get()[messageId]).find(
        reaction => reaction.author === 'user'
      )

      commitReaction(
        messageId,
        role,
        custom.rowId,
        reactions,
        mine?.emoji === DOUBLE_CLICK_REACTION ? null : DOUBLE_CLICK_REACTION
      )
    },
    [messageId, messageRuntime, role]
  )

  return enabled ? onDoubleClick : undefined
}
