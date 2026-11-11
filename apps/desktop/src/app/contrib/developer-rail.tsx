import { useStore } from '@nanostores/react'
import type { CSSProperties } from 'react'
import { useLocation } from 'react-router'

import { $narrowOverlayPaneIds } from '@/components/pane-shell/narrow-overlay-state'
import { $narrowViewport, $paneVisible, togglePaneVisible } from '@/components/pane-shell/tree/store'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $showsAdvancedChrome } from '@/store/interface-mode'
import { $rightRailActiveTabId, setFileBrowserOpen } from '@/store/layout'
import { $visiblePreviewTabs, browserTabFor, toggleBrowserTab } from '@/store/preview'
import { PREVIEW_TILE_PREFIX } from '@/store/preview-explicit'
import { toggleReview } from '@/store/review'
import { $focusedSessionIsTile, $focusedStoredSessionId } from '@/store/session-focus'
import { $focusedWorkspaceCwd } from '@/store/session-states'
import { usePortalShell } from '@/store/shell-variant'

import { hideDeveloperToolPanel, prepareDeveloperToolPanel } from '../redesign/developer-tool-panel'
import { toggleTerminalPane } from '../right-sidebar/terminal/reveal-focus'
import { appViewForPath } from '../routes'
import { useTitlebarHeight } from '../shell/use-touch-titlebar'

/** Only presentation lives here; the existing pane owners handle reveal and focus. */
export function DeveloperRail() {
  const redesigned = usePortalShell()
  const advanced = useStore($showsAdvancedChrome)
  const focusedChat = useStore($focusedSessionIsTile)
  const { pathname } = useLocation()

  return redesigned && advanced && (focusedChat || appViewForPath(pathname) === 'chat') ? <DeveloperTools /> : null
}

function DeveloperTools() {
  const { t } = useI18n()
  const titlebarHeight = useTitlebarHeight()
  const narrow = useStore($narrowViewport)
  const overlays = useStore($narrowOverlayPaneIds)
  const files = useStore($paneVisible('files'))
  const terminal = useStore($paneVisible('terminal'))
  const review = useStore($paneVisible('review'))
  const previewTabs = useStore($visiblePreviewTabs)
  const activeTabId = useStore($rightRailActiveTabId)
  const browserTab = browserTabFor(previewTabs, activeTabId)
  const browser = useStore($paneVisible(`${PREVIEW_TILE_PREFIX}:${browserTab?.id ?? ''}`))

  const tools = [
    {
      paneId: 'files',
      label: t.sidebar.files,
      icon: 'files',
      action: 'view.showFiles',
      active: narrow ? overlays.has('files') : files,
      run: () => (narrow ? setFileBrowserOpen(!overlays.has('files')) : togglePaneVisible('files'))
    },
    {
      paneId: 'terminal',
      label: t.sidebar.terminal,
      icon: 'terminal',
      action: 'view.showTerminal',
      active: terminal,
      run: toggleTerminalPane
    },
    {
      paneId: 'review',
      label: t.sidebar.review,
      icon: 'git-compare',
      action: 'view.toggleReview',
      active: narrow ? overlays.has('review') : review,
      run: () => {
        const tileId = $focusedSessionIsTile.get() && $focusedStoredSessionId.get()
        toggleReview(tileId ? $focusedWorkspaceCwd.get() || null : null, tileId ? `tile:${tileId}` : 'main')
      }
    },
    {
      paneId: `${PREVIEW_TILE_PREFIX}:${browserTab?.id ?? ''}`,
      label: t.preview.tab,
      icon: 'globe',
      action: 'view.showBrowser',
      active: browser,
      run: toggleBrowserTab
    }
  ]

  return (
    <nav
      aria-label={t.rightSidebar.panelsAria}
      className="portal-rail relative z-40 mt-(--titlebar-height) w-(--global-nav-width) shrink-0"
      data-slot="developer-rail"
      style={{ '--titlebar-height': `${titlebarHeight}px` } as CSSProperties}
    >
      <div className="portal-rail-group min-h-0 overflow-y-auto">
        {tools.map(tool => (
          <Tip
            key={tool.action}
            label={<TipKeybindLabel actionId={tool.action} text={tool.label} />}
            placement="right-rail"
          >
            <Button
              aria-label={tool.label}
              aria-pressed={tool.active}
              className="portal-chrome-button"
              onClick={() => {
                if (tool.active && hideDeveloperToolPanel(tool.paneId)) {
                  return
                }

                tool.run()

                // Existing browser tabs move now; newly registered tabs use their dock hint.
                const paneId =
                  tool.action === 'view.showBrowser'
                    ? `${PREVIEW_TILE_PREFIX}:${browserTabFor($visiblePreviewTabs.get(), $rightRailActiveTabId.get())?.id ?? ''}`
                    : tool.paneId

                prepareDeveloperToolPanel(paneId)
              }}
              size="icon"
              variant="ghost"
            >
              <Codicon name={tool.icon} />
            </Button>
          </Tip>
        ))}
      </div>
    </nav>
  )
}
