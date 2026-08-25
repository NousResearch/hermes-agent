import type { CSSProperties } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { useI18n } from '@/i18n'
import { windowBrowserTabId } from '@/store/windows'

import { PreviewTilePane } from './right-rail/preview'

/**
 * Dedicated shell for `?win=browser`: the in-app Browser, full-window, no
 * session sidebar or layout tree. Same webview + address bar the docked tab
 * uses — just in its own OS window.
 */
export function BrowserPopoutShell() {
  const { t } = useI18n()
  const tabId = windowBrowserTabId()

  return (
    <div
      className="flex h-screen min-h-0 w-screen flex-col bg-(--ui-bg-chrome) text-(--ui-text-primary)"
      data-contrib-shell=""
      style={{ '--titlebar-height': `${TITLEBAR_HEIGHT}px` } as CSSProperties}
    >
      <div
        aria-hidden="true"
        className="relative shrink-0 bg-(--ui-bg-chrome)"
        style={{ height: TITLEBAR_HEIGHT }}
      >
        {/* Same traffic-light / native-overlay carve-out as the main titlebar:
            a full-bar drag region would eat the window buttons. */}
        <div
          className="pointer-events-none absolute inset-y-0 left-0 w-(--titlebar-controls-left,14px) [-webkit-app-region:drag]"
        />
        <div
          className="pointer-events-none absolute inset-y-0 left-[calc(var(--titlebar-controls-left,14px)+(var(--titlebar-control-size,24px)*2)+0.75rem)] right-[calc(var(--titlebar-tools-right,0.75rem)+0.75rem)] [-webkit-app-region:drag]"
        />
      </div>
      <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
        {tabId ? (
          <PreviewTilePane tabId={tabId} />
        ) : (
          <div className="grid h-full place-items-center">
            <PanelEmpty description={t.preview.web.blankPageBody} icon="globe" />
          </div>
        )}
      </div>
    </div>
  )
}
