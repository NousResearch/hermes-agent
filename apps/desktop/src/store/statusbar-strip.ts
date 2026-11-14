import { computed } from 'nanostores'

import { $portalShell } from './shell-variant'

// A leaf of its own (not statusbar-prefs, whose mode store reads the window at import), so the
// keybind store can declare the statusbar toggle's availability without loading window state.

/** Whether this window's shell has a statusbar strip at all. The redesign has none: its rail and
 *  System popover carry the bar's items. Every way to toggle the bar (⌘⇧S, ⌘K, menus) offers it
 *  only here, so an invisible toggle can never hide the classic bar for a later rollback. */
export const $statusbarStripAvailable = computed($portalShell, portal => !portal)
