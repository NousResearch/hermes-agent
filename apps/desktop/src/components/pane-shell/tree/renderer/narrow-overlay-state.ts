import { atom } from 'nanostores'

interface NarrowOverlayChrome {
  paneId: string
  tabIds: readonly string[]
}

/** Actual mounted overlay chrome, not the persisted docked-open flags.
 * Lets titlebar affordances yield to their visible pane title on narrow. */
export const $narrowOverlayChrome = atom<NarrowOverlayChrome | null>(null)
