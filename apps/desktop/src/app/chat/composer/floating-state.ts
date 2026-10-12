import { atom } from 'nanostores'

import type { ComposerTarget } from './focus'

export interface FloatingComposerOwner {
  id: string
  groupId: string
  target: ComposerTarget
}

/** User intent, not whichever editor happened to receive a delayed DOM focus. */
export const $floatingComposerOwner = atom<FloatingComposerOwner | null>(null)
