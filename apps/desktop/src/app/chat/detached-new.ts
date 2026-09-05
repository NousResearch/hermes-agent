import { atom } from 'nanostores'

/** A detached chat (the workflow canvas, …) can claim `/new` so it remints
 *  its own thread instead of jumping to a workspace draft. Null = default. */
export const $detachedNewSession = atom<(() => void) | null>(null)
