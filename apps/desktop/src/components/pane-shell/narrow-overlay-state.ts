import { atom } from 'nanostores'

/** Transient visible overlay pane ids (including aliases and zone mates).
 * Docked/persisted open flags do not describe the narrow overlay. Ordered:
 * the revealed pane's ids first, its zone mates after, so `[...set][0]` names
 * the sheet on screen. */
export const $narrowOverlayPaneIds = atom<ReadonlySet<string>>(new Set())
