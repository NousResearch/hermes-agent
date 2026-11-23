import { atom } from 'nanostores'

/** Transient visible overlay pane ids (including aliases and zone mates).
 * Docked/persisted open flags do not describe the narrow overlay. */
export const $narrowOverlayPaneIds = atom<ReadonlySet<string>>(new Set())
