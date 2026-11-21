import type { ReactNode } from 'react'

/** Passive decorations inside ordinary conversation tool headers, not media or directives. */
export const TOOL_ACTIVITY_HEADER_AREA = 'chat.toolActivity.header'
export const TOOL_ACTIVITY_HEADER_API_VERSION = 1 as const

/** Frozen scalar-only presentation data. No arguments, output or execution controls. */
export interface ToolActivityHeaderContext {
  readonly toolName: string
  /** Inner name already known to the host for a bridged call. */
  readonly innerToolName?: string
  /** The native row's presentation status, not an independently inferred outcome. */
  readonly status: 'error' | 'notice' | 'running' | 'success' | 'warning'
  /** Whether the host has a result; interruption or status alone never implies one. */
  readonly hasResult: boolean
  readonly interrupted?: boolean
  /** Recorded Unix timestamps in seconds. Absent timing is never synthesized. */
  readonly startedAt?: number
  readonly completedAt?: number
  /** Reported terminal exit code, when available; it does not define status. */
  readonly exitCode?: number
  readonly hasDetails: boolean
  readonly expanded: boolean
}

/** `ctx.register({ id, area: TOOL_ACTIVITY_HEADER_AREA, data: { apiVersion, render } })`.
 * All compatible entries compose in registry order (ascending order, then registration).
 * Same-id registration replaces the entry; dispose before reusing an id, as in other areas.
 * A rendering exception blanks only that decoration until it is re-registered/remounted.
 * Native labels, grouping, timers, disclosure and output always remain host-owned.
 */
export interface ToolActivityHeaderContribution {
  readonly apiVersion: typeof TOOL_ACTIVITY_HEADER_API_VERSION
  /** Passive inline content only (no buttons, links, portals or DOM overrides).
   * Return null to decline. Mounted as a component, so hooks retain state on context updates.
   * Only runs while the native ordinary row is mounted; it never reveals collapsed groups.
   */
  readonly render: (context: ToolActivityHeaderContext) => ReactNode
}
