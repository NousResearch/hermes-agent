import { SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'

/** How many of an agent's chats its card shows before "Show all" (Var1 147:7470 draws 2–3). */
export const AGENT_CHAT_CAP = 3

/** The chats an agent card shows: its `limit` most recent, plus any further down that must stay in
 *  reach (the open chat, a running turn, unread news). Order is kept. */
export function capAgentChats<T>(rows: readonly T[], limit: number, keep: (row: T) => boolean): T[] {
  return rows.filter((row, index) => index < limit || keep(row))
}

/** The number an agent's card states for its chats. */
export interface AgentChatCount {
  value: number
  /** More chats exist than `value`: the card says "at least". */
  more: boolean
}

/** The agent's chats as its card states them. The backend's own count of the profile is the number
 *  (never under what is on screen: a chat can be listed before the count catches up). Without one (an
 *  older backend, or an agent on a machine the list was not read from) a profile whose list stopped at
 *  its page holds more chats than are loaded, and the loaded number moves as chats arrive and old
 *  ones leave the page (135, then 134): the card then states the full pages it holds as a floor
 *  ("100+") instead of passing the loaded window off as the total. */
export function agentChatCount(loaded: number, truncated: boolean, total?: number): AgentChatCount {
  if (total !== undefined) {
    return { value: Math.max(total, loaded), more: false }
  }

  if (!truncated || loaded < SIDEBAR_SESSIONS_PAGE_SIZE) {
    return { value: loaded, more: false }
  }

  return { value: loaded - (loaded % SIDEBAR_SESSIONS_PAGE_SIZE), more: true }
}
