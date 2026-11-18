import type { SessionInfo } from '@/hermes'
import type { AgentChatOrder } from '@/store/agent-sidebar-preferences'
import { sessionPinId } from '@/store/session'

const activity = (row: SessionInfo) => (row.last_active ?? row.started_at) * 1000

/** An agent's chats in the order the user arranged them. The arrangement holds for the chats it
 *  names until one of them has news: a chat active since then rises above it by recency, as a new
 *  chat does, so a hand-made order never buries what just happened. Chats it does not name (an
 *  older page that loaded later) stay below, by recency. Rows arrive most recent first. */
export function orderAgentChats(rows: SessionInfo[], order: AgentChatOrder | undefined): SessionInfo[] {
  if (!order?.ids.length) {
    return rows
  }

  const rank = new Map(order.ids.map((id, index) => [id, index]))
  const fresh: SessionInfo[] = []
  const arranged: SessionInfo[] = []
  const rest: SessionInfo[] = []

  for (const row of rows) {
    if (activity(row) > order.at) {
      fresh.push(row)
    } else {
      ;(rank.has(sessionPinId(row)) ? arranged : rest).push(row)
    }
  }

  arranged.sort((a, b) => rank.get(sessionPinId(a))! - rank.get(sessionPinId(b))!)

  return [...fresh, ...arranged, ...rest]
}

/** The arrangement a drag leaves: every chat of the agent in its new order, stamped no earlier than
 *  the latest of them was active (a gateway clock ahead of this one must not undo the drag). */
export function arrangeAgentChats(rows: readonly SessionInfo[], now = Date.now()): AgentChatOrder {
  return { at: Math.max(now, ...rows.map(activity)), ids: rows.map(row => sessionPinId(row)) }
}
