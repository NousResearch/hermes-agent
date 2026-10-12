import { computed } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'
import { $showsAdvancedChrome, modeBound } from '@/store/interface-mode'

// The colored profile strip at the sidebar foot. For someone who runs profiles
// as bots it duplicates the footer's gateway selector, so it can be switched
// off; while it is off the statusbar grows a profile dropdown beside the
// gateway switcher so switching profiles never loses its door. On by default.
// Simple mode rests it hidden (unless it is the only door left) without
// touching this preference.
const $profileRailVisiblePref = persistentAtom('hermes.desktop.profileRailVisible', true, Codecs.bool)

export const $profileRailVisible = modeBound('profileRailVisible', $profileRailVisiblePref, value =>
  $profileRailVisiblePref.set(value)
)

// The statusbar picker stands in for a hidden rail only in Advanced: Simple
// offers profiles from Organize chats and the composer.
export const $statusbarProfileSwitcherHidden = computed(
  [$profileRailVisible, $showsAdvancedChrome],
  (rail, advanced) => rail || !advanced
)

export function toggleProfileRailVisible() {
  $profileRailVisible.set(!$profileRailVisible.get())
}
