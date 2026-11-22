/**
 * A refreshed page and the window's live copy of the same stored rows, paired by the rows' own
 * identity (lib/chat-messages/live-row-identity).
 *
 * The backend emits a turn's interim narration, tool events and final reply naming the durable
 * row each one is (agent/live_row_identity.py). A live part whose row the page paints is that
 * row's copy: the page's part stays and the live one goes, whatever its words, its provider
 * tool-call id (`call_0` repeats within and across turns) or its place in the turn.
 *
 * A live part without an identity (the answer still streaming, an event from an older backend) is
 * never paired here; older backends keep the receipt path (transcript-receipt-turns).
 */

import {
  type ChatMessage,
  type ChatMessagePart,
  indexStoredParts,
  isLiveCopy,
  keepsItsLiveState
} from '@/lib/chat-messages'

type ToolCallPart = Extract<ChatMessagePart, { type: 'tool-call' }>

/** A stored call slot the page has no result for yet takes the live card's result. */
function withResultOf(stored: ToolCallPart, live: ToolCallPart): ToolCallPart {
  if (stored.result !== undefined || live.result === undefined) {
    return stored
  }

  return {
    ...stored,
    result: live.result,
    ...(live.isError !== undefined ? { isError: live.isError } : {}),
    ...(live.artifact !== undefined ? { artifact: live.artifact } : {}),
    ...(live.completedAt !== undefined ? { completedAt: live.completedAt } : {}),
    ...(live.toolResultMetadata !== undefined ? { toolResultMetadata: live.toolResultMetadata } : {})
  }
}

/**
 * The window without the parts the page stores, and the page with the results only the live cards
 * had. A window row whose every reply part the page stores goes (reasoning alone is no reply of its
 * own), unless it carries state the page cannot (an error, a stop, a part still streaming).
 */
export function reconcileLiveRowsWithPage(
  window: ChatMessage[],
  page: ChatMessage[]
): { window: ChatMessage[]; page: ChatMessage[] } {
  const stored = indexStoredParts(page)

  if (!stored.size) {
    return { window, page }
  }

  const patches = new Map<string, ToolCallPart>()
  const rows = new Map<number, ChatMessage | null>()

  window.forEach((message, at) => {
    if (message.role !== 'assistant') {
      return
    }

    // A cached fold from an earlier read keeps its parts under the stored-id merge's rules
    // (transcript-backfill); it only lends the page a result an earlier refresh filled in.
    const live = isLiveCopy(message)

    const parts = message.parts.filter(part => {
      const copy = stored.copyOf(part)

      if (!copy) {
        return true
      }

      const storedPart = page[copy.at].parts[copy.part]

      if (part.type === 'tool-call' && storedPart.type === 'tool-call') {
        const key = `${copy.at}:${copy.part}`
        const filled = withResultOf(patches.get(key) ?? storedPart, part)

        if (filled !== storedPart) {
          patches.set(key, filled)
        }
      }

      return !live
    })

    if (parts.length !== message.parts.length) {
      const kept = keepsItsLiveState(message) || parts.some(part => part.type !== 'reasoning')
      rows.set(at, kept ? { ...message, parts } : null)
    }
  })

  const nextPage = patches.size
    ? page.map((message, at) =>
        message.parts.some((_, part) => patches.has(`${at}:${part}`))
          ? { ...message, parts: message.parts.map((part, index) => patches.get(`${at}:${index}`) ?? part) }
          : message
      )
    : page

  const nextWindow = rows.size
    ? window.flatMap((message, at) => {
        const next = rows.has(at) ? rows.get(at) : message

        return next ? [next] : []
      })
    : window

  return { window: nextWindow, page: nextPage }
}
