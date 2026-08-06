import type { WindowPollingState } from './use-window-polling-state'

/**
 * Centralized polling intervals per feature and window state.
 *
 * State priority: hidden > idle > active
 * - null = stop polling entirely
 * - number = interval in milliseconds
 */
export const POLL_INTERVALS = {
  /** Background process monitoring in composer status stack */
  statusStack: { active: 5_000, idle: 15_000, hidden: null } as Record<WindowPollingState, number | null>,

  /** Gateway log tail in the gateway menu popover */
  gatewayLog: { active: 3_000, idle: null, hidden: null } as Record<WindowPollingState, number | null>,

  /** MCP server log tail in the MCP settings tab */
  mcpLog: { active: 2_000, idle: null, hidden: null } as Record<WindowPollingState, number | null>,

  /** Cron job run peek in the sidebar */
  cronPeek: {
    active: 8_000,
    idle: 8_000,
    hidden: 60_000,
  } as Record<WindowPollingState, number | null>,
} as const

export type PollFeature = keyof typeof POLL_INTERVALS

/**
 * Returns the polling interval for a feature in the given window state.
 * Returns null if polling should be stopped.
 */
export function getPollInterval(feature: PollFeature, state: WindowPollingState): number | null {
  return POLL_INTERVALS[feature][state] ?? null
}