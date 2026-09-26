import { atom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { modeBound } from '@/store/interface-mode'

/**
 * Whether the rail is a preference at all. The Webapp (browser-hosted Desktop)
 * always takes the statusbar profile picker, so its rail preference is fixed
 * off and the levers that would flip it hide rather than do nothing.
 */
export const PROFILE_RAIL_TOGGLEABLE = !isBrowserHostedDesktop()

// The colored profile strip at the sidebar foot. For someone who runs profiles
// as bots it duplicates the footer's gateway selector, so it can be switched
// off; while it is off the statusbar grows a profile dropdown beside the
// gateway switcher so switching profiles never loses its door. On by default,
// except in the Webapp, where the preference is the host's, not the user's.
// Simple mode rests it hidden (unless it is the only door left) without
// touching this preference — in the Webapp too, since Simple has no statusbar.
const $profileRailVisiblePref = PROFILE_RAIL_TOGGLEABLE
  ? persistentAtom('hermes.desktop.profileRailVisible', true, Codecs.bool)
  : atom(false)

export const $profileRailVisible = modeBound('profileRailVisible', $profileRailVisiblePref, value => {
  if (PROFILE_RAIL_TOGGLEABLE) {
    $profileRailVisiblePref.set(value)
  }
})

export function toggleProfileRailVisible() {
  $profileRailVisible.set(!$profileRailVisible.get())
}
