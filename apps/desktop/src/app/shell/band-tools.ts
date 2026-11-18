import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'

import { $widthClass } from '@/app/redesign/breakpoints'
import { $narrowViewport } from '@/components/pane-shell/tree/store'
import { $chromiumAppWindow } from '@/lib/keybinds/app-window'
import { canUseHud } from '@/store/hud'
import { $interfaceMode } from '@/store/interface-mode'
import { $leftSideOpen, $sidebarOpen } from '@/store/layout'
import { $portalShell } from '@/store/shell-variant'
import type { BandToolId } from '@/store/titlebar-app-actions'

import { $workspaceIsPage } from '../routes'

import { bandOffersHistory } from './history-position'

/** Where a wide redesigned window offers the way back to a hidden chat sidebar: Advanced in the band,
 *  Simple as the floating button at the stage's corner (PanelRevealButton). The open Chats rail button
 *  folds it (Figma 1054:20985); pages have no sidebar to bring back. The floating button names the
 *  chats, so it follows the sidebar itself; the band's is positional. */
export const $sidebarRevealSurface = computed(
  [$portalShell, $narrowViewport, $workspaceIsPage, $interfaceMode, $sidebarOpen, $leftSideOpen],
  (portal, narrow, workspacePage, mode, sidebarOpen, leftSideOpen): 'band' | 'floating' | null => {
    if (!portal || narrow || workspacePage) {
      return null
    }

    if (mode === 'simple') {
      return sidebarOpen ? null : 'floating'
    }

    return leftSideOpen ? null : 'band'
  }
)

/** The band's fixed tools hidden in this window's state. TitlebarControls renders from it and the
 *  titlebar width reservation counts from it, so a hidden tool never holds a phantom slot. A hook,
 *  not a computed store: the host facts it reads (HUD support, browser hosting) are not stores. */
export function useBandHiddenTools(): ReadonlySet<BandToolId> {
  const portal = useStore($portalShell)
  const compact = useStore($widthClass) === 'compact'
  const narrow = useStore($narrowViewport)
  const workspacePage = useStore($workspaceIsPage)
  const mode = useStore($interfaceMode)
  const reveal = useStore($sidebarRevealSurface)
  const appWindow = useStore($chromiumAppWindow)
  const portalCompact = portal && compact
  const noHistory = !portal || !bandOffersHistory(compact, appWindow)

  // Below wide the sidebar button is the band's menu: the chat sidebar as a sheet, and on a
  // compact page (no rail) the rail as one; a medium page keeps its rail and needs none. A wide
  // band carries it only where the reveal surface is the band.
  const hidden: Record<BandToolId, boolean> = {
    'agent-settings': !portal || workspacePage,
    'flip-panes': portalCompact,
    'history-back': noHistory,
    'history-forward': noHistory,
    hud: !canUseHud(),
    layout: portal && (mode === 'simple' || portalCompact),
    'right-sidebar': false,
    settings: portal,
    sidebar: portal && (narrow ? workspacePage && !compact : reveal !== 'band')
  }

  return new Set((Object.keys(hidden) as BandToolId[]).filter(id => hidden[id]))
}
