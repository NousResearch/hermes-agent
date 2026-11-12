import { useAuiState, useMessageRuntime } from '@assistant-ui/react'
import { registryBackendScopeKey } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { type MouseEvent, useCallback, useRef } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import type { ChatMessage } from '@/lib/chat-messages'
import { triggerHaptic } from '@/lib/haptics'
import { isTouchInteraction } from '@/lib/touch-interaction'
import { activeGatewayConnectionId } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { QUICK_REACTIONS, toggleMessageReaction } from '@/store/reactions'
import { $reactionsEnabled } from '@/store/reactions-enabled'
import {
  $agentReactions,
  $localReactions,
  agentLiveReactions,
  mergeReactions,
  setLocalReaction
} from '@/store/reactions-local'
import { sessionEventScopeFor } from '@/store/session-states'
import type { MessageReaction } from '@/types/hermes'

// Stable empty identity — a fresh [] per render would re-run every consumer.
const EMPTY_REACTIONS: MessageReaction[] = []

/** The tapback a double-click lands: Apple's first Tapback, and ours. */
export const DOUBLE_CLICK_REACTION = QUICK_REACTIONS[0]

// Double-click means something else on these: links and controls act, inputs
// and code blocks select. The gesture only claims plain message body.
const NOT_A_TAPBACK = 'a, button, input, pre, select, textarea, [contenteditable="true"], [role="button"]'

/**
 * A press that ends further than this from where it went down was a DRAG, not a
 * tap. macOS double-click slop is a couple of pixels; a selection drag is tens
 * to hundreds.
 */
const TAPBACK_SLOP_PX = 5

export interface TapbackGesture {
  /** Release point, straight off the MouseEvent. */
  clientX?: number
  clientY?: number
  detail: number
  /** Where this gesture's press went down; ignored when absent. */
  press?: null | { x: number; y: number }
  target: EventTarget | null
}

/**
 * Is this double-click the "heart it" gesture?
 *
 * `detail === 2` keeps a triple-click (select-the-paragraph) from re-firing,
 * and anything the browser already gives a double-click meaning keeps it.
 *
 * A press that ENDS somewhere other than where it went down was a drag-select:
 * the press picked the word and the drag extended the highlight. Claiming that
 * as a tapback is what used to eat the selection, because the handler clears it
 * (a real tapback leaves a stray highlighted word otherwise). A gesture that
 * travelled down the page belongs to the text, so it is not a tapback.
 */
export function isTapbackDoubleClick(event: TapbackGesture): boolean {
  if (event.detail !== 2) {
    return false
  }

  const target = event.target

  if (target instanceof Element && target.closest(NOT_A_TAPBACK)) {
    return false
  }

  const { press } = event

  if (press && event.clientX !== undefined && event.clientY !== undefined) {
    if (Math.hypot(event.clientX - press.x, event.clientY - press.y) > TAPBACK_SLOP_PX) {
      return false
    }
  }

  return true
}

/** Paint the tapback locally, then persist behind it. */
function commitReaction(
  messageId: string,
  role: ChatMessage['role'],
  rowId: number | undefined,
  reactions: MessageReaction[],
  emoji: null | string,
  sessionId: null | string
): void {
  // Flip the UI immediately — a tapback is direct manipulation and must never
  // wait on a round-trip. Persistence follows in the background.
  setLocalReaction(messageId, emoji)
  void toggleMessageReaction({ id: messageId, role, rowId, reactions } as ChatMessage, emoji, 'user', sessionId)
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
  const agentAll = useStore($agentReactions)
  const sessionView = useSessionView()
  const runtimeSessionId = useStore(sessionView.$runtimeId)
  const storedSessionId = useStore(sessionView.$storedId)
  const sessionId = runtimeSessionId ?? storedSessionId

  // The agent overlay is keyed by bare DB row id, and row ids are only
  // meaningful within ONE source's database. Resolve the source this
  // displayed session actually belongs to — the scope its own events
  // proved, falling back to the actively served source when the runtime's
  // events arrived untagged (local legacy primary) — so an overlay recorded
  // on source A can never repaint a coincidental same-numbered row on
  // source B (a tile from another connection, a cross-source resume).
  const activeProfile = useStore($activeGatewayProfile)

  const viewScope =
    sessionEventScopeFor(runtimeSessionId) ?? registryBackendScopeKey(activeGatewayConnectionId(), activeProfile)

  const agentLive = rowId === undefined ? undefined : agentLiveReactions(agentAll, rowId, viewScope)

  return {
    enabled,
    react: useCallback(
      (emoji: null | string) => commitReaction(messageId, role, rowId, reactions, emoji, sessionId),
      [messageId, reactions, role, rowId, sessionId]
    ),
    reactions: mergeReactions(reactions, localAll[messageId], agentLive)
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
 *
 * Two handlers, not one: `dblclick` reports only where the gesture ENDED, and a
 * double-click that ends somewhere else was a drag-select (press picks the word,
 * drag extends the highlight). The press point is recorded per message element,
 * so a gesture that travelled is left to the text instead of being hearted and
 * having its selection cleared.
 */
export function useTapbackDoubleClick(
  messageId: string,
  role: ChatMessage['role']
): { onDoubleClick: (event: MouseEvent<HTMLElement>) => void; onMouseDown: (event: MouseEvent<HTMLElement>) => void } | undefined {
  const enabled = useStore($reactionsEnabled)
  const messageRuntime = useMessageRuntime()
  const sessionView = useSessionView()
  const runtimeSessionId = useStore(sessionView.$runtimeId)
  const storedSessionId = useStore(sessionView.$storedId)
  const sessionId = runtimeSessionId ?? storedSessionId
  const press = useRef<null | { x: number; y: number }>(null)

  const onMouseDown = useCallback((event: MouseEvent<HTMLElement>) => {
    press.current = { x: event.clientX, y: event.clientY }
  }, [])

  const onDoubleClick = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (
        isTouchInteraction(event.nativeEvent) ||
        !isTapbackDoubleClick({
          clientX: event.clientX,
          clientY: event.clientY,
          detail: event.detail,
          press: press.current,
          target: event.target
        })
      ) {
        return
      }

      // Double-click has already selected the word underneath — the tapback,
      // not a stray selection, is what the gesture meant. A gesture that dragged
      // never reaches here (the press/release check above), so this only ever
      // clears the single-word pick a real tapback leaves behind.
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
        mine?.emoji === DOUBLE_CLICK_REACTION ? null : DOUBLE_CLICK_REACTION,
        sessionId
      )
    },
    [messageId, messageRuntime, role, sessionId]
  )

  return enabled ? { onDoubleClick, onMouseDown } : undefined
}
