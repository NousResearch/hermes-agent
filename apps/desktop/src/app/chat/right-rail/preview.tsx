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
import { PANE_TAB_STRIP_LINE, PaneTab, PaneTabLabel } from '@/components/ui/pane-tab'
import { Tip } from '@/components/ui/tooltip'
import { translateNow, useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
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
  closeOtherRightRailTabs,
  closeRightRail,
  closeRightRailTab,
  closeRightRailTabsToRight,
  type PreviewTarget
} from '@/store/preview'
import { $dirtyPreviewUrls } from '@/store/preview-edit'

import { PreviewPane } from './preview-pane'

export const PREVIEW_RAIL_MIN_WIDTH = '18rem'
export const PREVIEW_RAIL_MAX_WIDTH = '38rem'

interface ChatPreviewRailProps {
  onRestartServer?: (url: string, context?: string) => Promise<string>
  setTitlebarToolGroup?: SetTitlebarToolGroup
}

interface RailTab {
  id: RightRailTabId
  label: string
  target: PreviewTarget
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
  const previewTarget = useStore($previewTarget)
  const dirtyPreviewUrls = useStore($dirtyPreviewUrls)

  const tabs = useMemo<readonly RailTab[]>(
    () => [
      ...(previewTarget
        ? [{ id: RIGHT_RAIL_PREVIEW_TAB_ID, label: t.preview.tab, target: previewTarget } as RailTab]
        : []),
      ...filePreviewTabs.map(({ id, target }) => ({ id, label: tabLabelFor(target), target }) as RailTab)
    ],
    [filePreviewTabs, previewTarget, t.preview.tab]
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
      <div
        className={cn(
          'group/rail-tabs flex h-(--titlebar-height) shrink-0 bg-(--ui-sidebar-surface-background)',
          PANE_TAB_STRIP_LINE
        )}
      >
        <div
          className="flex min-w-0 flex-1 overflow-x-auto overflow-y-hidden overscroll-x-contain [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          role="tablist"
        >
          {tabs.map((tab, index) => {
            const active = tab.id === activeTab.id
            const hasOthers = tabs.length > 1
            const hasTabsToRight = index < tabs.length - 1
            const dirty = Boolean(dirtyPreviewUrls[tab.target.url])

            return (
              <ContextMenu key={tab.id}>
                <ContextMenuTrigger asChild>
                  <PaneTab active={active} dirty={dirty} onClose={() => closeRightRailTab(tab.id)}>
                    <Tip label={tab.target.path || tab.target.url || tab.label}>
                      <PaneTabLabel
                        aria-selected={active}
                        as="button"
                        className="normal-case tracking-normal"
                        onClick={() => selectRightRailTab(tab.id)}
                        role="tab"
                        type="button"
                      >
                        {tab.label}
                      </PaneTabLabel>
                    </Tip>
                  </PaneTab>
                </ContextMenuTrigger>
                <ContextMenuContent>
                  <ContextMenuItem onSelect={() => closeRightRailTab(tab.id)}>
                    {t.common.close}
                    <span className="ml-auto pl-4 text-(--ui-text-tertiary)">{formatCombo('mod+w')}</span>
                  </ContextMenuItem>
                  <ContextMenuItem disabled={!hasOthers} onSelect={() => closeOtherRightRailTabs(tab.id)}>
                    {t.preview.closeOthers}
                  </ContextMenuItem>
                  <ContextMenuItem disabled={!hasTabsToRight} onSelect={() => closeRightRailTabsToRight(tab.id)}>
                    {t.preview.closeToRight}
                  </ContextMenuItem>
                  <ContextMenuSeparator />
                  <ContextMenuItem onSelect={closeRightRail}>{t.preview.closeAll}</ContextMenuItem>
                </ContextMenuContent>
              </ContextMenu>
            )
          })}
        </div>
        <button
          aria-label={t.preview.closePane}
          className="mr-1.5 grid size-6 shrink-0 self-center place-items-center rounded-md text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring group-hover/rail-tabs:opacity-100 [-webkit-app-region:no-drag]"
          onClick={closeRightRail}
          type="button"
        >
          <Codicon name="close" size="0.75rem" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        <PreviewPane
          embedded
          onRestartServer={isPreview ? onRestartServer : undefined}
          reloadRequest={previewReloadRequest}
          setTitlebarToolGroup={setTitlebarToolGroup}
          target={activeTab.target}
        />
      </div>
    </aside>
  )
}
