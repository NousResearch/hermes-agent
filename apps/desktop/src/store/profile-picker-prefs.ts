import { Codecs, persistentAtom } from '@/lib/persisted'

// App-global renderer presentation state, like session-list density. The key
// deliberately excludes connection/profile identity so switching either keeps
// the chosen picker mode; localStorage restores it on reload and app restart.
const KEY = 'hermes.desktop.global.profilePicker.alwaysUseDropdown'

export const $alwaysUseProfileDropdown = persistentAtom(KEY, false, Codecs.bool)

export function setAlwaysUseProfileDropdown(enabled: boolean) {
  $alwaysUseProfileDropdown.set(enabled)
}
