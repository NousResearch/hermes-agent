/**
 * Roadmaps plugin — embedded configuration. Thresholds, machine-state labels,
 * tones and the module tabs are declared once here so views stay data-driven.
 * English on purpose: the UI labels the state machine, it never rewrites it.
 */

export const ID = 'roadmaps'

/** Responsive layout thresholds (measured on the real container). */
export const LAYOUT = {
  compact: 900,
  wide: 1280,
  inspectorWidth: 340
} as const

/** Query refresh intervals (ms) — polls, with invalidation as the fast path. */
export const QUERY = {
  listRefetchMs: 30_000,
  plansRefetchMs: 30_000,
  snapshotRefetchMs: 60_000
} as const

/** Semantic StatusDot tone per node lifecycle state. */
export type Tone = 'good' | 'muted' | 'warn' | 'bad'

export const NODE_TONE: Record<string, Tone> = {
  ready: 'good',
  in_progress: 'warn',
  blocked: 'bad',
  completed: 'muted',
  cancelled: 'muted'
}

/** Thread sort order — blocked first, then in flight, then ready. */
export const NODE_ORDER: Record<string, number> = {
  blocked: 0,
  in_progress: 1,
  ready: 2
}

/** Machine-state labels (English, config-driven). */
export const NODE_STATE_LABEL: Record<string, string> = {
  ready: 'Ready',
  in_progress: 'In progress',
  blocked: 'Blocked',
  completed: 'Completed',
  planned: 'Planned',
  archived: 'Archived'
}

/** Label for each next-action kind the copilot can surface. */
export const NEXT_ACTION_LABEL: Record<string, string> = {
  unblock: 'Unblock',
  claim: 'Claim',
  advance: 'Advance',
  assign: 'Assign',
  'wait-deps': 'Wait'
}

/** Canonical relation labels + icons. */
export const RELATION_LABEL: Record<string, string> = {
  depends_on: 'depends on',
  blocks: 'blocks'
}

export const RELATION_ICON: Record<string, string> = {
  depends_on: 'arrow-right',
  blocks: 'debug-disconnect'
}

/** The module navigation tabs (order = display order). */
export const TABS: Array<{ id: string; label: string; codicon: string }> = [
  { id: 'thread', label: 'Thread', codicon: 'list-ordered' },
  { id: 'map', label: 'Map', codicon: 'graph' },
  { id: 'plan', label: 'Plan', codicon: 'versions' },
  { id: 'milestones', label: 'Milestones', codicon: 'milestone' },
  { id: 'decisions', label: 'Decisions', codicon: 'checklist' },
  { id: 'files', label: 'Files', codicon: 'files' }
]

/** Tabs that participate in node selection + the Inspector panel. */
export const INSPECTOR_TABS = new Set(['thread', 'map', 'milestones'])
