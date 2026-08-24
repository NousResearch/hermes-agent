/**
 * Which tip comes next.
 *
 * The catalog is a ring and the rotation walks it in order, resuming from
 * whichever tip was last shown. Order and not shuffle: the catalog reads as a
 * tour of the app — the rail down the left, then the composer, then the pane on
 * the right — and walking it that way means a user who sees three tips over a
 * week sees three neighbouring parts of the app, not three unrelated ones.
 *
 * Two rules on top of the walk:
 *
 * 1. A hard close RETIRES a tip. It is gone for good — the only way back is
 *    Settings → Reset. Retiring every tip is a legitimate end state: the
 *    rotation runs dry and the app stops talking, which is what a user who
 *    closed all of them was asking for.
 * 2. A tip is skipped, not waited for, when it has nothing on screen to point
 *    at. `available` is the subset that resolves right now, and the walk steps
 *    over the rest — but it counts position against the FULL catalog, so which
 *    panes happen to be open changes what you see and never the order you see
 *    it in.
 */

export interface TipRotationState {
  /** The tip shown most recently, retired or not. Where the next walk starts. */
  lastShownId: null | string
  /** Hard-closed tip ids. */
  retired: readonly string[]
}

/**
 * The first tip after `lastShownId` that is live and on screen, wrapping at the
 * end of the catalog. Null when the rotation is spent.
 *
 * @param order Every tip id, in catalog order — the ring being walked.
 * @param available The subset with something on screen to point at.
 */
export function nextTip(
  order: readonly string[],
  available: readonly string[],
  state: TipRotationState
): null | string {
  // An unknown or absent last tip starts the walk at the top of the catalog.
  const start = state.lastShownId == null ? -1 : order.indexOf(state.lastShownId)

  for (let step = 1; step <= order.length; step += 1) {
    const id = order[(start + step + order.length) % order.length]

    if (available.includes(id) && !state.retired.includes(id)) {
      return id
    }
  }

  return null
}
