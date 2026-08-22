import type { ReactNode } from 'react'

/**
 * Where a contribution came from. `'core'` is the app's own default UI;
 * anything else is a plugin/extension id (e.g. `'plugin:kanban'`). This is the
 * provenance tag that drives precedence and, later, the trust/capability gate
 * (WoW-style taint: plugin-sourced contributions can be blocked from privileged
 * actions unless granted).
 */
export type ContributionSource = 'core' | (string & {})

/**
 * Live boolean authority for contribution visibility. This is intentionally a
 * structural contract rather than a Nano Stores import: core and plugin state
 * can supply any readonly source with the same `get` + `listen` semantics.
 *
 * The registry owns the listener lifetime. A visibility source therefore
 * invalidates its projected area when it changes, and is detached when the
 * contribution is replaced or removed.
 */
export interface ContributionCondition {
  get(): boolean
  listen(listener: (value: boolean) => void): () => void
}

/**
 * The single, uniform primitive every surface consumes. A bar renders these as
 * inline items via `<Slot>`; a dock renders them as stacked/tabbed panes via
 * `<PaneHost>`. Same shape either way -- the host decides how to present them.
 */
export interface Contribution {
  /** Stable id, unique within its area. Re-registering the same id replaces it. */
  id: string
  /** Namespaced area id this contribution targets, e.g. `'secondarySidebar'`. */
  area: string
  /** Provenance; defaults to `'core'` when omitted. */
  source?: ContributionSource
  /** Human label (pane tab / header). Optional for bar items. */
  title?: string
  /** Ascending sort key within the area; ties keep insertion order. */
  order?: number
  /** Live visibility authority. Omit for always-on. The registry subscribes to
   *  this source and invalidates exactly this contribution's area on change. */
  when?: ContributionCondition
  /** Soft disable without unregistering. `false` hides it. */
  enabled?: boolean
  /** Renders the contribution's content (UI contributions). */
  render?: () => ReactNode
  /**
   * Declarative payload for data contributions (Family B): layout presets,
   * themes, commands — anything consumed by an engine rather than rendered.
   */
  data?: unknown
}
