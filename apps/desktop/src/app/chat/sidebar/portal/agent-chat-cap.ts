import { SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'

/** How many of an agent's chats its card shows before "Show all" (Var1 147:7470 draws 2–3). */
export const AGENT_CHAT_CAP = 3

/** Row and chrome heights (px) used to distribute spare vertical sidebar room before folding. */
const CHAT_ROW_PX = 30
const AGENT_HEADER_PX = 56
const MACHINE_HEADER_PX = 34
const SECTION_HEADER_PX = 34

export interface AgentChatBudgetInput {
  /** Measured height of the sidebar's scrollable sections container (px). 0 falls back to AGENT_CHAT_CAP. */
  availableHeight: number
  /** Fixed non-agent rows above the agent list (pinned chats + group chats + their section headers). */
  pinnedRows: number
  groupRows: number
  machineCount: number
  collapsedAgentCount: number
  /** Open (expanded) agents in display order. */
  openAgents: readonly {
    key: string
    count: number
    showAll?: boolean
    active?: boolean
  }[]
}

/**
 * Dynamically allocates per-agent chat caps so open agents fill available sidebar height before
 * folding behind "Show all". Every open agent keeps at least `AGENT_CHAT_CAP` rows; surplus rows
 * water-fill across agents with more chats (active agent first on odd remainders), and an agent is
 * never folded for a single lone overflow chat (which would trade 1 chat row for 1 button row).
 */
export function computeAgentChatLimits(input: AgentChatBudgetInput): Record<string, number> {
  const limits: Record<string, number> = {}

  for (const agent of input.openAgents) {
    limits[agent.key] = AGENT_CHAT_CAP
  }

  if (input.availableHeight <= 0 || input.openAgents.length === 0) {
    return limits
  }

  const pinnedPx = input.pinnedRows > 0 ? SECTION_HEADER_PX + input.pinnedRows * CHAT_ROW_PX : SECTION_HEADER_PX
  const groupPx = input.groupRows > 0 ? SECTION_HEADER_PX + input.groupRows * CHAT_ROW_PX : 0
  const agentsChromePx =
    SECTION_HEADER_PX +
    input.machineCount * MACHINE_HEADER_PX +
    (input.openAgents.length + input.collapsedAgentCount) * AGENT_HEADER_PX

  let budgetRows = Math.floor((input.availableHeight - pinnedPx - groupPx - agentsChromePx) / CHAT_ROW_PX)

  // Reserve every open agent's baseline rows (full count if already expanded, else min(count, floor)).
  for (const agent of input.openAgents) {
    const baseline = agent.showAll ? agent.count : Math.min(agent.count, AGENT_CHAT_CAP)
    // Reserve 1 row for "Show all" when this agent still hides > 1 chat.
    const foldRow = !agent.showAll && agent.count > baseline + 1 ? 1 : 0
    budgetRows -= baseline + foldRow
  }

  // Water-fill any surplus height across capped agents that still have hidden chats.
  const capped = [
    ...input.openAgents.filter(a => !a.showAll && a.active),
    ...input.openAgents.filter(a => !a.showAll && !a.active)
  ]

  while (budgetRows > 0) {
    let grew = false

    for (const agent of capped) {
      if (budgetRows <= 0) {
        break
      }

      if (limits[agent.key] < agent.count) {
        limits[agent.key] += 1
        budgetRows -= 1
        grew = true
      }
    }

    if (!grew) {
      break
    }
  }

  // Never hide a single lone chat behind a "Show all" button (same 1-row height cost).
  for (const agent of capped) {
    if (agent.count === limits[agent.key] + 1) {
      limits[agent.key] = agent.count
    }
  }

  return limits
}

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
