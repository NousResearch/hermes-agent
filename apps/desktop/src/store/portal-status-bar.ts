import { computed } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

import { modeBound } from './interface-mode'
import { $portalShell } from './shell-variant'

// The redesign's status bar (Figma "status bar", rev 11/16: 1262:39850 open, 1264:40699 collapsed):
// the readouts that describe the chat being worked in, as chips in the canvas under the workspace
// frame, with a toggle at their end. The rest of the classic bar stays in the rail's System popover.
// Its own preference, so collapsing it never touches the classic bar's visibility.
const $portalStatusBarPref = persistentAtom('hermes.desktop.portalStatusBar.v1', true, Codecs.bool)

// Open by default with developer chrome only: Simple rests it collapsed, and its toggle there opens
// it for the session (interface-mode.ts).
const $portalStatusBarChoice = modeBound('portalStatusBarOpen', $portalStatusBarPref, open =>
  $portalStatusBarPref.set(open)
)

/** The bar is open on screen: the redesigned shell, and the mode-resolved choice. */
export const $portalStatusBarOpen = computed([$portalShell, $portalStatusBarChoice], (portal, open) => portal && open)

export function setPortalStatusBarOpen(open: boolean) {
  $portalStatusBarChoice.set(open)
}
