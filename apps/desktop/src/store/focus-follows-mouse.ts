import { Codecs, persistentAtom } from '@/lib/persisted'

/** A machine-level interaction preference, not session or composer placement state. */
export const $focusFollowsMouse = persistentAtom('hermes.desktop.focusFollowsMouse', false, Codecs.bool)

export function setFocusFollowsMouse(enabled: boolean) {
  $focusFollowsMouse.set(enabled)
}
