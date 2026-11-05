import type { RosterRow } from './types'

/**
 * The bot a Screen command names, resolved against whatever roster is current at
 * CLICK time.
 *
 * Resolving late is the point: a row captured when the palette was registered (a
 * launch that never opened the Bots pane gets its rows from `profiles.list`) carries
 * no connection binding, and a Screen pane opened with one builds its ws URL with a
 * null connection id — it connects, receives nothing, and paints black. The Bots
 * pane's own snapshot is what the pane's routing is proven on, so the command asks
 * for a row again when it runs instead of holding one.
 *
 * A bot's name is its identity here (one row per name); with several connections
 * listing the same name, the first row wins, which is the active connection's.
 */
export function pickScreenBot(rows: readonly RosterRow[] | null | undefined, name: string): RosterRow | null {
  const target = String(name || '').trim()

  if (!target || !Array.isArray(rows)) {
    return null
  }

  return rows.find(row => row?.name === target) || null
}
