import type { ReactNode } from 'react'

/**
 * Session-row decoration surface — the seams a plugin can decorate a sidebar
 * session row through, with the SAME registry schema as every other surface
 * (statusbar, composer, panes):
 *
 *   render areas (`data`):  sessionRow.leading   — inline right after the
 *                                                  status dot / drag handle
 *                             sessionRow.trailing  — inline before the row's
 *                                                  hover actions cluster
 *
 * Core keeps ownership of the row's layout, gestures, and labels — these seams
 * AUGMENT a row with a small decoration (a badge, a colour swatch, a tag), they
 * never replace it. A contribution renders `null` for rows it doesn't own, so
 * registering one costs nothing on every other row in the list.
 */

export const SESSION_ROW_AREAS = {
  leading: 'sessionRow.leading',
  trailing: 'sessionRow.trailing'
} as const

/** Props handed to a session-row decoration's `render`. */
export interface SessionRowSlotProps {
  /** The STORED (durable) id of the session the row renders — the lineage root,
   *  not the live id. Auto-compression rotates the live id, so a plugin that
   *  remembers `session.id` decorates the row until the next compaction and then
   *  silently stops matching; the durable id is the one `host.sessions.*` and
   *  core's own pin/reorder address (see `sessionPinId`). */
  sessionId: string
}

/** Payload of a `sessionRow.*` contribution's `data`. */
export interface SessionRowSlotContribution {
  /** Renders the decoration, or `null` to leave the row untouched. Mounted as
   *  a component inside the contribution error boundary, so it can subscribe
   *  to its own stores; a throw degrades to an inline error, not a dead row. */
  render: (props: SessionRowSlotProps) => ReactNode
}
