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

import { type ChatMessage, indexStoredParts, lendLiveResults, liveRemainder } from '@/lib/chat-messages'

/**
 * The window without the parts the page stores, and the page with the results only the live cards
 * had (`liveRemainder`). A cached fold from an earlier read keeps its parts under the stored-id
 * merge's rules (transcript-backfill); it only lends the page a result an earlier refresh filled in.
 */
export function reconcileLiveRowsWithPage(
  window: ChatMessage[],
  page: ChatMessage[]
): { window: ChatMessage[]; page: ChatMessage[] } {
  const stored = indexStoredParts(page)

  if (!stored.size) {
    return { window, page }
  }

  return { window: window.flatMap(message => liveRemainder(stored, message)), page: lendLiveResults(page, window) }
}
