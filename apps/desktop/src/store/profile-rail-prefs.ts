import { Codecs, persistentAtom } from '@/lib/persisted'

// The colored profile strip at the sidebar foot. For someone who runs profiles
// as bots it duplicates the footer's gateway selector, so it can be switched
// off; while it is off the statusbar grows a profile dropdown beside the
// gateway switcher so switching profiles never loses its door. On by default.
export const $profileRailVisible = persistentAtom('hermes.desktop.profileRailVisible', true, Codecs.bool)

export function toggleProfileRailVisible() {
  $profileRailVisible.set(!$profileRailVisible.get())
}
