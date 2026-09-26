/**
 * Which tip comes next.
 *
 * The catalog is a ring and the rotation walks it in order, resuming from
 * whichever tip was last shown. Order and not shuffle: the catalog reads as a
 * tour of the app — the rail down the left, then the composer, then the pane on
 * the right — and walking it that way means a user who sees three tips over a
 * week sees three neighbouring parts of the app, not three unrelated ones.
 *
 * Three rules on top of the walk:
 *
 * 1. The walk is ONE lap. A tip that has been on screen — whether the timer
 *    took it down or the user did — is not offered again; the catalog is an
 *    introduction, and an introduction repeated is a nag. Once every tip has
 *    had its moment the rotation runs dry and the app stops talking. Settings
 *    → Reset is the only way to a second lap. (Agent tips carry no catalog id
 *    and never enter this ledger.)
 * 2. A hard close RETIRES a tip, which is the same silence made explicit —
 *    it also survives a Reset of nothing else, so the ✕ stays the heavier
 *    gesture.
 * 3. A tip is skipped, not waited for, when it has nothing on screen to point
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
  /** Every tip id that has been on screen, however it left. */
  seen: readonly string[]
}

/**
 * The first tip after `lastShownId` that is unseen, live and on screen, wrapping
 * at the end of the catalog. Null when the rotation is spent.
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

    if (available.includes(id) && !state.retired.includes(id) && !state.seen.includes(id)) {
      return id
    }
  }

  return null
}
