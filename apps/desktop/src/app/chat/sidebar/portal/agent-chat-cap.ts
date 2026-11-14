/** How many of an agent's chats its card shows before "Show all" (Var1 147:7470 draws 2–3). */
export const AGENT_CHAT_CAP = 3

/** The chats an agent card shows: its `limit` most recent, plus any further down that must stay in
 *  reach (the open chat, a running turn, unread news). Order is kept. */
export function capAgentChats<T>(rows: readonly T[], limit: number, keep: (row: T) => boolean): T[] {
  return rows.filter((row, index) => index < limit || keep(row))
}

/** The `$sidebarAgentExpanded` entry that holds an agent's "Show all" choice: its own key, scoped by
 *  the agent's machine and profile like the disclosure beside it. */
export const agentShowAllKey = (agentKey: string) => `all:${agentKey}`
