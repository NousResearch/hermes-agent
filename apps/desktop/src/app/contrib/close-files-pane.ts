import { dismissTreePane, paneRootSide } from '@/components/pane-shell/tree/store'
import { fileBrowserSide, setFileBrowserOpen } from '@/store/layout'

import { inDeveloperToolPanel } from '../redesign/developer-tool-panel'

/**
 * Files' Close. In its root side column it collapses that side, which keeps ⌘J and the titlebar
 * button truthful. Dragged next to main, a side collapse cannot hide it (the collapse skips
 * main-bearing children), so Close dismisses it there, or ⌘W would silently do nothing.
 *
 * The redesign's tool panel is that side too, whatever else stands in it: a Browser tab registers
 * as a main surface, so the pane tree stops reading the panel's zone as the right side once one
 * joins (developer-tool-panel.ts).
 */
export function closeFilesPane(): void {
  if (paneRootSide('files') === fileBrowserSide() || inDeveloperToolPanel('files')) {
    setFileBrowserOpen(false)
  } else {
    dismissTreePane('files')
  }
}
