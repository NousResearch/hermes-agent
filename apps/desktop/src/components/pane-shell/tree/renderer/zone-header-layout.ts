import { TITLEBAR_HEIGHT, TOUCH_TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { useTouchTitlebar } from '@/app/shell/use-touch-titlebar'

import { isSessionStripPane } from '../store'

/** The desktop tab strip's row (`h-7`). */
const TAB_STRIP_HEIGHT = 28

interface ZoneHeaderLayoutInput {
  /** The titlebar measured too little room beside its controls for the tabs. */
  measuredBelowControls: boolean
  narrow: boolean
  shown: readonly string[]
  /** No workspace or main pane in the zone. */
  sidebarGroup: boolean
  topEdge: boolean
}

export interface ZoneHeaderLayout {
  /** A narrow touch zone holding session tabs swaps the scrolling strip for
   *  the compact picker. */
  compactTabs: boolean
  tabStripHeight: number
  /** A top-edge zone's tabs take their own row under the window controls. */
  tabsBelowControls: boolean
  /** A top-edge zone's tabs share the titlebar band with the controls. */
  tabsInTitlebar: boolean
  titlebarHeight: number
}

/** Where a zone's tabs sit relative to the window titlebar, and how tall the
 *  titlebar band and the tab row are. */
export function useZoneHeaderLayout({
  measuredBelowControls,
  narrow,
  shown,
  sidebarGroup,
  topEdge
}: ZoneHeaderLayoutInput): ZoneHeaderLayout {
  const touchTitlebar = useTouchTitlebar()
  const compactTabs = touchTitlebar && narrow && shown.some(isSessionStripPane)
  // The compact picker needs a full touch row: the titlebar's fit threshold
  // also has to accommodate the drag handle and the separate new-tab target.
  const tabsBelowControls = topEdge && (compactTabs || sidebarGroup || measuredBelowControls)

  return {
    compactTabs,
    // The compact tab row is a touch target the same height as the touch band.
    tabStripHeight: compactTabs ? TOUCH_TITLEBAR_HEIGHT : TAB_STRIP_HEIGHT,
    tabsBelowControls,
    tabsInTitlebar: topEdge && !tabsBelowControls,
    titlebarHeight: touchTitlebar ? TOUCH_TITLEBAR_HEIGHT : TITLEBAR_HEIGHT
  }
}
