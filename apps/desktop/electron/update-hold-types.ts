// What the blocked boot screen shows while an update hold keeps the local
// backend stopped (R8 D3). Shared by the main process (update-hold-wiring.ts)
// and the renderer (src/global.d.ts), so the wire shape has one definition.
export interface UpdateHoldWire {
  holdId: string
  /** `held`: a leftover process holds the checkout lock; `busy`/`error`: ownership could not be verified. */
  verdict: 'held' | 'busy' | 'error'
  /** The update process that wrote the marker (exited), when known. */
  ownerPid: number | null
  /** Epoch ms the hold was first seen. */
  since: number
  /** Epoch ms of the latest check. */
  checkedAt: number
  logPath: string
}
