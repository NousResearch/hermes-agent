import { useStore } from '@nanostores/react'
import { type CSSProperties, useEffect } from 'react'

import { $interfaceMode, shownInMode } from '@/store/interface-mode'
import { $connection } from '@/store/session'
import { $titlebarAppActionsSide, titlebarAppActionsClusterCounts } from '@/store/titlebar-app-actions'

import { useWindowControlsOverlayInsets } from '../../shell/hooks/use-window-controls-overlay-width'
import {
  TITLEBAR_CHROME_CHANGED_EVENT,
  titlebarControlsPosition,
  titlebarToolsRightCss,
  titlebarToolsWidthCss,
  TOUCH_TITLEBAR_HEIGHT
} from '../../shell/titlebar'
import { useTouchTitlebar } from '../../shell/use-touch-titlebar'
import { useTitlebarToolContributions } from '../panes'

// Where the fixed titlebar clusters sit (the same vars AppShell sets). Browser
// touch chrome pins them flush in its 44px band, at touch-target size.
function useTitlebarControlsVars(controlsPos: { left: number; top: number }): CSSProperties {
  const touchTitlebar = useTouchTitlebar()

  return {
    '--titlebar-controls-left': `${controlsPos.left}px`,
    '--titlebar-controls-top': `${touchTitlebar ? 0 : controlsPos.top}px`,
    ...(touchTitlebar
      ? {
          '--titlebar-control-size': `${TOUCH_TITLEBAR_HEIGHT}px`,
          '--titlebar-control-height': `${TOUCH_TITLEBAR_HEIGHT}px`
        }
      : {})
  } as CSSProperties
}

// The REAL titlebar tool clusters (sidebar/flip toggles, haptics, keybinds,
// settings gear) — fixed chrome positioned via the same CSS vars AppShell
// sets, computed here from the live connection. Page-registered tools
// (preview's monitor/devtools cluster, …) arrive as registry contributions.
export function useTitlebarChromeLayout() {
  const leftTitlebarTools = useTitlebarToolContributions('left')
  const rightTitlebarTools = useTitlebarToolContributions('right')
  const connection = useStore($connection)
  // Windows/Linux reserve native min/max/close in the WCO overlay. Prefer the
  // live measurement — it also reports WHICH side the host placed them on
  // (KDE/GTK button layouts can put them left); the static reservation only
  // applies when the WCO API is unavailable.
  const overlayInsets = useWindowControlsOverlayInsets()
  const nativeOverlayWidth = overlayInsets?.right ?? connection?.nativeOverlayWidth ?? 0
  const nativeLeftInset = overlayInsets?.left ?? 0

  const controlsPos = titlebarControlsPosition(
    connection?.windowButtonPosition,
    Boolean(connection?.isFullscreen),
    nativeLeftInset
  )

  const controlsVars = useTitlebarControlsVars(controlsPos)

  const titlebarChrome = {
    darwinMajor: connection?.darwinMajor ?? 0,
    isFullscreen: Boolean(connection?.isFullscreen),
    windowButtonPosition: connection?.windowButtonPosition
  }

  const titlebarToolsRight = titlebarToolsRightCss(nativeOverlayWidth, titlebarChrome)

  // WSLg: Electron's native overlay drifts its hit-region under RAIL, so the
  // renderer paints its own min/max/close (main decides via customWindowControls).
  const customWindowControls =
    connection?.customWindowControls ?? window.hermesDesktop?.windowControls?.custom ?? false

  const appActionsSide = useStore($titlebarAppActionsSide)
  const interfaceMode = useStore($interfaceMode)
  const shownTool = shownInMode(interfaceMode)
  const paneToolCount = rightTitlebarTools.filter(tool => !tool.hidden && shownTool(tool)).length
  const leftExtraCount = leftTitlebarTools.filter(tool => !tool.hidden && shownTool(tool)).length
  const clusters = titlebarAppActionsClusterCounts(appActionsSide, leftExtraCount, 0, interfaceMode)
  const systemToolsWidth = titlebarToolsWidthCss(clusters.right)

  const titlebarToolsWidth =
    paneToolCount > 0 ? `calc(${systemToolsWidth} + ${titlebarToolsWidthCss(paneToolCount)})` : systemToolsWidth

  const leftToolsWidth = titlebarToolsWidthCss(clusters.left)

  // Native caption reservations can translate chrome without resizing it.
  useEffect(() => {
    window.dispatchEvent(new CustomEvent(TITLEBAR_CHROME_CHANGED_EVENT))
  }, [controlsPos.left, controlsPos.top, titlebarToolsRight])

  return {
    controlsVars,
    customWindowControls,
    leftTitlebarTools,
    leftToolsWidth,
    rightTitlebarTools,
    systemToolsWidth,
    titlebarChrome,
    titlebarToolsRight,
    titlebarToolsWidth
  }
}
