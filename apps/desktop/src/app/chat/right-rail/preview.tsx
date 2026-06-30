import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import type { SetTitlebarToolGroup } from '@/app/shell/titlebar-controls'
import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { Tip } from '@/components/ui/tooltip'
import { translateNow, useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import {
  $browserEnabled,
  $browserTabs,
  type BrowserTabState,
  clearBrowserTabs,
  closeBrowserTab,
  enableBrowserAndOpenTab,
  isBrowserTabId,
  moveBrowserTab,
  openBrowserTab
} from '@/store/browser'
import {
  $panesFlipped,
  $rightRailActiveTabId,
  RIGHT_RAIL_PREVIEW_TAB_ID,
  type RightRailTabId,
  selectRightRailTab
} from '@/store/layout'
import {
  $filePreviewTabs,
  $previewReloadRequest,
  $previewTarget,
  closeRightRail,
  closeRightRailTab,
  type PreviewTarget
} from '@/store/preview'
import { $dirtyPreviewUrls } from '@/store/preview-edit'
import { $activeSessionId } from '@/store/session'

import { BrowserPane } from './browser-pane'
import { PreviewPane } from './preview-pane'

export const PREVIEW_RAIL_MIN_WIDTH = '18rem'
export const PREVIEW_RAIL_MAX_WIDTH = '38rem'

const INTRINSIC = `clamp(${PREVIEW_RAIL_MIN_WIDTH}, 36vw, 32rem)`

// Track for <Pane id="preview">. Folds the intrinsic clamp with a min-floor
// against --chat-min-width so the chat surface never gets squeezed below it.
// Subtracts the project browser width so preview yields rather than crushing
// the chat when both right-side panes are open.
export const PREVIEW_RAIL_PANE_WIDTH = `min(${INTRINSIC}, max(0rem, calc(100vw - var(--pane-chat-sidebar-width) - var(--pane-file-browser-width, 0rem) - var(--chat-min-width))))`

interface ChatPreviewRailProps {
  onRestartServer?: (url: string, context?: string) => Promise<string>
  setTitlebarToolGroup?: SetTitlebarToolGroup
}

type RailTab =
  | {
      id: RightRailTabId
      kind: 'preview'
      label: string
      target: PreviewTarget
    }
  | {
      id: RightRailTabId
      kind: 'browser'
      label: string
      tab: BrowserTabState
    }

function tabLabelFor(target: PreviewTarget): string {
  const value = target.label || target.path || target.source || target.url
  const tail = value.split(/[\\/]/).filter(Boolean).at(-1)

  return tail || value || translateNow('preview.tab')
}

export function ChatPreviewRail({ onRestartServer, setTitlebarToolGroup }: ChatPreviewRailProps) {
  const { t } = useI18n()
  const previewReloadRequest = useStore($previewReloadRequest)
  const activeTabId = useStore($rightRailActiveTabId)
  const panesFlipped = useStore($panesFlipped)
  const filePreviewTabs = useStore($filePreviewTabs)
  const browserTabs = useStore($browserTabs)
  const browserEnabled = useStore($browserEnabled)
  const activeSessionId = useStore($activeSessionId)
  const previewTarget = useStore($previewTarget)
  const dirtyPreviewUrls = useStore($dirtyPreviewUrls)

  const tabs = useMemo<readonly RailTab[]>(
    () => [
      ...(previewTarget
        ? [
            {
              id: RIGHT_RAIL_PREVIEW_TAB_ID,
              kind: 'preview' as const,
              label: t.preview.tab,
              target: previewTarget
            } as RailTab
          ]
        : []),
      ...filePreviewTabs.map(
        ({ id, target }) => ({ id, kind: 'preview' as const, label: tabLabelFor(target), target }) as RailTab
      ),
      ...(browserEnabled
        ? browserTabs.map(tab => ({ id: tab.id, kind: 'browser' as const, label: tab.title || tab.url, tab }) as RailTab)
        : [])
    ],
    [browserTabs, browserEnabled, filePreviewTabs, previewTarget, t.preview.tab]
  )

  const activeTab = tabs.find(tab => tab.id === activeTabId) ?? tabs[0]

  useEffect(() => {
    if (activeTab && activeTab.id !== activeTabId) {
      selectRightRailTab(activeTab.id)
    }
  }, [activeTab, activeTabId])

  if (!activeTab) {
    return null
  }

  const isPreview = activeTab.id === RIGHT_RAIL_PREVIEW_TAB_ID

  const closeTab = (tab: RailTab) => {
    if (tab.kind === 'browser' && isBrowserTabId(tab.id)) {
      closeBrowserTab(tab.id)
    } else {
      closeRightRailTab(tab.id)
    }
  }

  const closeOtherTabs = (keepId: RightRailTabId) => {
    for (const tab of tabs) {
      if (tab.id !== keepId) {
        closeTab(tab)
      }
    }

    selectRightRailTab(keepId)
  }

  const closeTabsToRight = (tabId: RightRailTabId) => {
    const index = tabs.findIndex(tab => tab.id === tabId)

    if (index === -1) {
      return
    }

    for (const tab of tabs.slice(index + 1)) {
      closeTab(tab)
    }
  }

  const closeAllTabs = () => {
    closeRightRail()
    clearBrowserTabs()
  }

  const openExternalUrl = (url: string) => {
    if (window.hermesDesktop?.openExternal) {
      void window.hermesDesktop.openExternal(url)

      return
    }

    window.open(url, '_blank', 'noopener,noreferrer')
  }

  const enableAndOpenBrowserTab = () => {
    enableBrowserAndOpenTab({ sessionId: activeSessionId })
  }

  const tabTip = (tab: RailTab): string => (tab.kind === 'browser' ? tab.tab.url : tab.target.path || tab.target.url || tab.label)

  return (
    <aside
      className={cn(
        'relative flex h-full w-full min-w-0 flex-col overflow-hidden border-(--ui-stroke-tertiary) bg-(--ui-editor-surface-background) text-(--ui-text-tertiary)',
        panesFlipped ? 'border-r' : 'border-l'
      )}
      // Windows/WSLg paint Electron's Window Controls Overlay across our
      // titlebar band, so the editor-style tab strip (which normally sits IN that
      // band) would land under the fixed titlebar tools. --right-rail-top-inset
      // (set by AppShell only when the overlay is present) drops the rail one
      // titlebar-height so it opens below the band. 0px elsewhere → unchanged.
      style={{ paddingTop: 'var(--right-rail-top-inset, 0px)' }}
    >
      <div className="group/rail-tabs flex h-(--titlebar-height) shrink-0 border-b border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background)">
        <div
          className="flex min-w-0 flex-1 overflow-x-auto overflow-y-hidden overscroll-x-contain [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          role="tablist"
        >
          {tabs.map((tab, index) => {
            const active = tab.id === activeTab.id
            const hasOthers = tabs.length > 1
            const hasTabsToRight = index < tabs.length - 1
            const dirty = tab.kind === 'preview' ? Boolean(dirtyPreviewUrls[tab.target.url]) : false
            const browserIndex = tab.kind === 'browser' ? browserTabs.findIndex(browserTab => browserTab.id === tab.id) : -1

            return (
              <ContextMenu key={tab.id}>
                <ContextMenuTrigger asChild>
                  <div
                    className={cn(
                      'group/tab relative flex h-full min-w-0 max-w-48 shrink-0 items-center text-[0.6875rem] font-medium [-webkit-app-region:no-drag] last:border-r last:border-(--ui-stroke-quaternary)',
                      active
                        ? 'bg-(--ui-editor-surface-background) text-foreground [--tab-bg:var(--ui-editor-surface-background)]'
                        : 'border-r border-(--ui-stroke-quaternary) text-(--ui-text-tertiary) [--tab-bg:var(--ui-sidebar-surface-background)] hover:bg-(--chrome-action-hover) hover:text-foreground'
                    )}
                    // Middle-click closes the tab, matching browser/IDE muscle
                    // memory. `onMouseDown` swallows the middle-button press so
                    // Chromium doesn't switch into autoscroll mode.
                    onAuxClick={event => {
                      if (event.button !== 1) {
                        return
                      }

                      event.preventDefault()
                      closeTab(tab)
                    }}
                    onMouseDown={event => {
                      if (event.button === 1) {
                        event.preventDefault()
                      }
                    }}
                  >
                    {active && (
                      <span aria-hidden="true" className="absolute inset-x-0 top-0 h-px bg-(--ui-stroke-primary)" />
                    )}
                    <Tip label={tabTip(tab)}>
                      <button
                        aria-selected={active}
                        className="flex h-full min-w-0 max-w-full items-center overflow-hidden pl-3 pr-2 text-left outline-none"
                        onClick={() => selectRightRailTab(tab.id)}
                        role="tab"
                        type="button"
                      >
                        <span className="block min-w-0 truncate">{tab.label}</span>
                      </button>
                    </Tip>
                    <span
                      aria-hidden="true"
                      className="pointer-events-none absolute inset-y-0 right-0 w-9 bg-[linear-gradient(to_right,transparent,var(--tab-bg)_55%)] opacity-0 transition-opacity group-hover/tab:opacity-100 group-focus-within/tab:opacity-100"
                    />
                    {dirty && (
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute right-1.5 top-1/2 grid size-4 -translate-y-1/2 place-items-center opacity-100 transition-opacity group-hover/tab:opacity-0 group-focus-within/tab:opacity-0"
                      >
                        {/* Amber (our warn color); a tab-bg ring + soft drop keeps it
                            legible where it overlaps the filename. */}
                        <span className="size-2 rounded-full bg-amber-500 shadow-[0_0_0_2px_var(--tab-bg),0_1px_2px_rgba(0,0,0,0.45)] dark:bg-amber-400" />
                      </span>
                    )}
                    <button
                      aria-label={t.preview.closeTab(tab.label)}
                      className="pointer-events-none absolute right-1.5 top-1/2 grid size-4 -translate-y-1/2 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 transition-[background-color,color,opacity] hover:bg-(--ui-bg-secondary) hover:text-foreground focus-visible:pointer-events-auto focus-visible:opacity-100 group-hover/tab:pointer-events-auto group-hover/tab:opacity-100 group-focus-within/tab:pointer-events-auto group-focus-within/tab:opacity-100"
                      onClick={() => closeTab(tab)}
                      type="button"
                    >
                      <Codicon name="close" size="0.75rem" />
                    </button>
                  </div>
                </ContextMenuTrigger>
                <ContextMenuContent>
                  <ContextMenuItem onSelect={() => closeTab(tab)}>
                    {t.common.close}
                    <span className="ml-auto pl-4 text-(--ui-text-tertiary)">{formatCombo('mod+w')}</span>
                  </ContextMenuItem>
                  <ContextMenuItem disabled={!hasOthers} onSelect={() => closeOtherTabs(tab.id)}>
                    {t.preview.closeOthers}
                  </ContextMenuItem>
                  <ContextMenuItem disabled={!hasTabsToRight} onSelect={() => closeTabsToRight(tab.id)}>
                    {t.preview.closeToRight}
                  </ContextMenuItem>
                  {tab.kind === 'browser' ? (
                    <>
                      <ContextMenuSeparator />
                      <ContextMenuItem disabled={browserIndex <= 0} onSelect={() => moveBrowserTab(tab.tab.id, 'left')}>
                        Move tab left
                      </ContextMenuItem>
                      <ContextMenuItem
                        disabled={browserIndex === -1 || browserIndex >= browserTabs.length - 1}
                        onSelect={() => moveBrowserTab(tab.tab.id, 'right')}
                      >
                        Move tab right
                      </ContextMenuItem>
                      <ContextMenuItem onSelect={() => openExternalUrl(tab.tab.url)}>Open externally</ContextMenuItem>
                    </>
                  ) : null}
                  <ContextMenuSeparator />
                  <ContextMenuItem onSelect={closeAllTabs}>{t.preview.closeAll}</ContextMenuItem>
                </ContextMenuContent>
              </ContextMenu>
            )
          })}
        </div>
        {browserEnabled ? (
          <button
            aria-label="New Browser tab"
            className="grid size-6 shrink-0 self-center place-items-center rounded-md text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring group-hover/rail-tabs:opacity-100 [-webkit-app-region:no-drag]"
            onClick={() => openBrowserTab({ sessionId: activeSessionId })}
            type="button"
          >
            <Codicon name="globe" size="0.75rem" />
          </button>
        ) : (
          <button
            aria-label="Enable Browser"
            className="grid size-6 shrink-0 self-center place-items-center rounded-md text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring group-hover/rail-tabs:opacity-100 [-webkit-app-region:no-drag]"
            onClick={enableAndOpenBrowserTab}
            type="button"
          >
            <Codicon name="globe" size="0.75rem" />
          </button>
        )}
        <button
          aria-label={t.preview.closePane}
          className="mr-1.5 grid size-6 shrink-0 self-center place-items-center rounded-md text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring group-hover/rail-tabs:opacity-100 [-webkit-app-region:no-drag]"
          onClick={closeAllTabs}
          type="button"
        >
          <Codicon name="close" size="0.75rem" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        {activeTab.kind === 'browser' ? (
          <BrowserPane tab={activeTab.tab} />
        ) : (
          <PreviewPane
            embedded
            onRestartServer={isPreview ? onRestartServer : undefined}
            reloadRequest={previewReloadRequest}
            setTitlebarToolGroup={setTitlebarToolGroup}
            target={activeTab.target}
          />
        )}
      </div>
    </aside>
  )
}
