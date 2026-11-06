import type { ScopeGuard } from './middleware'

export interface ChatProbe {
  /** The chat the user is in now (stored id; null on a fresh chat). */
  current: () => null | string
  /** Subscribe to changes; returns the unsubscribe. */
  listen: (listener: () => void) => () => void
}

/** Holds a review to the chat the user pressed Enter in. The baseline is read
 *  from the same source it is later compared with, so id-domain differences
 *  (lineage root vs live tip) can never read as a switch. A real switch calls
 *  `onDrift` once and leaves `stillCurrent` false. */
export function createScopeGuard(probe: ChatProbe, onDrift: () => void): ScopeGuard {
  const start = probe.current()
  const stillCurrent = () => probe.current() === start

  const release = probe.listen(() => {
    if (!stillCurrent()) {
      onDrift()
    }
  })

  return { stillCurrent, release }
}
