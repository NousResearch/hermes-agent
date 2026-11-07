import { Codecs, persistentAtom } from '@/lib/persisted'

/**
 * A machine-level interaction preference, not session or composer placement
 * state. On by default: it restates the documented pointer behaviour, and the
 * switch is an opt-out for readers whose pointer crosses another chat on the
 * way to Files. Off means only a click or the keyboard selects a chat.
 */
export const $focusFollowsMouse = persistentAtom('hermes.desktop.focusFollowsMouse', true, Codecs.bool)

export function setFocusFollowsMouse(enabled: boolean) {
  $focusFollowsMouse.set(enabled)
}
