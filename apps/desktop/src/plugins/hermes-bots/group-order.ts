interface OrderRow {
  activity: number
  kind: 'bot' | 'group'
  name?: string
  pinned: boolean
  /** Manual position within the pinned band. Undefined = never hand-placed,
   *  keeps activity order (sinks below placed rows, same as a group room
   *  without `rosterOrder`). */
  order?: number
}

/**
 * Reorder room slots, not bots or folders. Pinning remains the outer band.
 *
 * Manual `order` reorders a row among ITS OWN KIND's slots — a group room's
 * `rosterOrder` among rooms, a bot's meta `rosterOrder` among bots — so a
 * hand-placed room never drags a bot row around, and vice versa. Rows without
 * an order keep their activity slot; a band nobody has hand-placed sorts
 * exactly as it did before ordering existed.
 */
export function sortGroupRosterRows<T extends OrderRow>(
  rows: T[],
  rooms: Record<string, { rosterOrder?: number }>
): T[] {
  const legacy = rows.slice().sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.activity - a.activity)
  const orderOf = (row: T) =>
    row.order ?? (row.kind === 'group' ? rooms[row.name!]?.rosterOrder : undefined) ?? Infinity

  const slotsOf = (kind: 'bot' | 'group') =>
    legacy
      .filter(row => row.kind === kind)
      .sort((a, b) => Number(b.pinned) - Number(a.pinned) || orderOf(a) - orderOf(b))

  const groups = slotsOf('group')
  const bots = slotsOf('bot')
  let groupIndex = 0
  let botIndex = 0

  return legacy.map(row => (row.kind === 'group' ? groups[groupIndex++] : bots[botIndex++]))
}

/** Swap visible neighbours without dropping filtered-out rows from the order. */
export function reorderGroupRows(rows: OrderRow[], name: string, delta: -1 | 1, visible?: string[]): string[] | null {
  const row = rows.find(row => row.name === name)

  const band = rows.filter(
    candidate =>
      candidate.kind === 'group' && candidate.pinned === row?.pinned && (!visible || visible.includes(candidate.name!))
  )

  const index = band.findIndex(candidate => candidate.name === name)
  const neighbour = index >= 0 ? band[index + delta] : undefined

  if (!neighbour) {
    return null
  }

  return rows
    .filter(row => row.kind === 'group')
    .map(row => (row.name === name ? neighbour.name! : row.name === neighbour.name ? name : row.name!))
}

/**
 * Insertion order for a drag-drop: the new roster-order value for every row of
 * the dragged bot's band, or null when the drop is a no-op. `bandRows` are the
 * bot rows of ONE section (or the flat list), in current display order, with
 * the same-band guarantee the menu path has (`pinned` equal, `kind` 'bot').
 *
 * Returns name → rosterOrder covering the WHOLE band, not just the two swapped
 * neighbours: insertion is not a swap, and partial writes would leave the
 * persisted orders unable to reproduce the on-screen sequence.
 */
export function insertBotOrderByDrop(
  bandRows: Array<{ name: string; pinned: boolean }>,
  dragName: string,
  targetName: string,
  after: boolean
): Record<string, number> | null {
  if (dragName === targetName) {
    return null
  }

  const names = bandRows.map(row => row.name)
  const from = names.indexOf(dragName)
  const target = names.indexOf(targetName)

  if (from < 0 || target < 0) {
    return null
  }

  // Guard the band guarantee: a pinned bot never drags among unpinned and
  // vice versa (pinning stays the outer band, same contract as rooms).
  if (bandRows[from]!.pinned !== bandRows[target]!.pinned) {
    return null
  }

  const next = names.slice()
  next.splice(from, 1)
  const insertAt = next.indexOf(targetName) + (after ? 1 : 0)
  next.splice(insertAt, 0, dragName)

  const assignments: Record<string, number> = {}
  next.forEach((name, index) => {
    assignments[name] = index
  })

  return assignments
}
