import { SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'

/** Baseline floor of chats an open agent shows before folding (Var1 147:7470 draws 2–3). */
export const AGENT_CHAT_CAP = 3
/** Stepped expansion chunk for "Show more"; snap straight to all when <= 12 remain. */
export const AGENT_CHAT_STEP = 10
export const AGENT_CHAT_SNAP_REMAINING = 12
/** Maximum extra unread rows kept above the fold when an agent is folded. */
export const MAX_FOLDED_UNREAD_KEEP = 2

/** Rendered row and chrome heights (px) measured from live Chromium layout so water-fill uses
 *  real spare viewport height without overflowing sibling agents off-screen. */
const CHAT_ROW_PX = 40
const AGENT_CARD_CHROME_PX = 69
const COLLAPSED_AGENT_PX = 63
const MACHINE_HEADER_PX = 40
const SECTION_HEADER_PX = 36
const EMPTY_PINNED_PX = 56
const BOT_SCREEN_ROW_PX = 156
const SAFETY_BUFFER_PX = 8

export interface AgentChatBudgetInput {
  /** Measured height of the sidebar's scrollable sections container (px). 0 falls back to AGENT_CHAT_CAP. */
  availableHeight: number
  /** Direct measured height of the Agents scroll viewport (`.portal-section-scroll`), when mounted.
   *  When > 0, replaces Pinned/Group-chat chrome estimation with the exact live viewport height. */
  agentsViewportHeight?: number
  /** Exact measured extra chrome inside open agents (e.g. visible Bot Screen thumbnails, px). */
  extraChromePx?: number
  /** Fixed non-agent rows above the agent list (pinned chats + group chats + their section headers). */
  pinnedRows: number
  groupRows: number
  machineCount: number
  collapsedAgentCount: number
  /** Visible Bot Screen live thumbnails inside open agent cards (used when `extraChromePx` is omitted). */
  botScreenRows?: number
  /** Open (expanded) agents in display order. */
  openAgents: readonly {
    key: string
    count: number
    showAll?: boolean
    active?: boolean
  }[]
}

/** Compute how many additional chats one "Show more" click should reveal. */
export function computeShowMoreStep(remaining: number): number {
  return remaining <= AGENT_CHAT_SNAP_REMAINING ? remaining : AGENT_CHAT_STEP
}

function computeUsableAgentPx(input: AgentChatBudgetInput): number {
  const screenPx = input.extraChromePx ?? (input.botScreenRows ?? 0) * BOT_SCREEN_ROW_PX
  const perAgentChromePx =
    input.machineCount * MACHINE_HEADER_PX +
    input.openAgents.length * AGENT_CARD_CHROME_PX +
    input.collapsedAgentCount * COLLAPSED_AGENT_PX +
    screenPx +
    SAFETY_BUFFER_PX

  if ((input.agentsViewportHeight ?? 0) > 0) {
    return input.agentsViewportHeight! - perAgentChromePx
  }

  const pinnedPx = input.pinnedRows > 0 ? SECTION_HEADER_PX + input.pinnedRows * CHAT_ROW_PX : EMPTY_PINNED_PX
  const groupPx = input.groupRows > 0 ? SECTION_HEADER_PX + input.groupRows * CHAT_ROW_PX : 0

  return input.availableHeight - pinnedPx - groupPx - SECTION_HEADER_PX - perAgentChromePx
}

/**
 * Dynamically allocates per-agent resting chat caps using real rendered chrome heights so open
 * agents water-fill true spare viewport room without pushing sibling agents below the fold:
 * - Every open agent keeps at least `AGENT_CHAT_CAP` (3) rows.
 * - Spare vertical rows water-fill round-robin across agents with hidden chats (active first) with
 *   NO arbitrary ceiling — a single agent or tall monitor uses every available row cleanly, while
 *   accurate chrome/Bot-Screen/project subtraction keeps all sibling agents visible above the fold.
 * - Never hides a single lone overflow chat behind a fold row (same 1-row height cost).
 */
export function computeAgentChatLimits(input: AgentChatBudgetInput): Record<string, number> {
  const limits: Record<string, number> = {}

  for (const agent of input.openAgents) {
    limits[agent.key] = AGENT_CHAT_CAP
  }

  const hasViewport = (input.agentsViewportHeight ?? 0) > 0 || input.availableHeight > 0

  if (!hasViewport || input.openAgents.length === 0) {
    return limits
  }

  let budgetRows = Math.floor(computeUsableAgentPx(input) / CHAT_ROW_PX)
  // Reserve every open agent's baseline rows (full count if already expanded, else min(count, floor)).
  for (const agent of input.openAgents) {
    const baseline = agent.showAll ? agent.count : Math.min(agent.count, AGENT_CHAT_CAP)
    const foldRow = !agent.showAll && agent.count > baseline + 1 ? 1 : 0
    budgetRows -= baseline + foldRow
  }

  // Water-fill true spare viewport height round-robin across capped agents (active agent first).
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

  // Never hide a single lone chat behind a "Show more" / "Show all" button (same 1-row height cost).
  for (const agent of capped) {
    if (agent.count === limits[agent.key] + 1) {
      limits[agent.key] = agent.count
    }
  }

  return limits
}
/** The chats an agent card shows: its `limit` most recent, plus any further down that must stay in
 *  reach (selected or working chat unconditionally; extra unread rows bounded by `maxSoftKeep` so
 *  unread cron/background bursts never blow past the cap). Order is kept. */
export function capAgentChats<T>(
  rows: readonly T[],
  limit: number,
  keep: (row: T) => boolean,
  options?: { softKeep?: (row: T) => boolean; maxSoftKeep?: number }
): T[] {
  const softKeep = options?.softKeep
  const maxSoftKeep = options?.maxSoftKeep ?? MAX_FOLDED_UNREAD_KEEP
  let softKept = 0

  return rows.filter((row, index) => {
    if (index < limit || keep(row)) {
      return true
    }

    if (softKeep && softKept < maxSoftKeep && softKeep(row)) {
      softKept += 1

      return true
    }

    return false
  })
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
