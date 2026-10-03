import { useStore } from '@nanostores/react'
import { type CSSProperties, useEffect, useRef } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { hiddenPaneProps, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { useActiveTabVisible } from '@/components/pane-shell/tree/renderer/tab-strip-scroll'
import { Codicon } from '@/components/ui/codicon'
import { PaneStripGlyph, PaneTab, PaneTabLabel, PaneTabStrip } from '@/components/ui/pane-tab'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $browserWorkspaces, commandBrowserWorkspace, installBrowserWorkspaceSync } from '@/store/browser-workspaces'
import { $bindings, bindingsFor } from '@/store/keybinds'
import { $browserPages } from '@/store/preview'
import { windowBrowserTabId, windowBrowserWorkspaceId } from '@/store/windows'

import { BROWSER_TAB_ACTIONS } from '../../../electron/browser-tab-shortcuts'

import { runBrowserTabAction } from './browser-tab-actions'
import { PreviewTilePane } from './right-rail/preview'
import { installPopoutPreviewResponder } from './right-rail/preview-popout-bridge'

/** One native workspace, keyed guests. Visibility is not guest lifecycle. */
export function BrowserPopoutShell() {
  const { t } = useI18n()
  const windowId = windowBrowserWorkspaceId()
  const states = useStore($browserWorkspaces)
  const pages = useStore($browserPages)
  const state = windowId ? states[windowId] : null
  const legacyTabId = !windowId ? windowBrowserTabId() : null
  const activeId = state?.activeTabId ?? ''
  const strip = useRef<HTMLDivElement>(null)

  useActiveTabVisible(strip, activeId, {
    enabled: Boolean(state),
    last: state?.tabs.at(-1)?.id === activeId,
    tabCount: state?.tabs.length ?? 0
  })
  useEffect(() => installBrowserWorkspaceSync(), [])
  useEffect(() => installPopoutPreviewResponder(), [])
  useEffect(() => {
    const api = window.hermesDesktop?.browserWorkspace

    if (!api || !windowId) {
      return
    }

    const stop = $bindings.subscribe(() =>
      api.setShortcuts(Object.fromEntries(BROWSER_TAB_ACTIONS.map(action => [action, bindingsFor(action)])))
    )

    const stopShortcut = api.onShortcut(runBrowserTabAction)

    return () => {
      stop()
      stopShortcut()
    }
  }, [windowId])

  return (
    <div
      className="flex h-screen min-h-0 w-screen flex-col bg-(--ui-bg-chrome) text-(--ui-text-primary)"
      data-contrib-shell=""
      style={{ '--titlebar-height': `${TITLEBAR_HEIGHT}px` } as CSSProperties}
    >
      <div aria-hidden="true" className="relative shrink-0 bg-(--ui-bg-chrome)" style={{ height: TITLEBAR_HEIGHT }}>
        <div className="pointer-events-none absolute inset-y-0 left-0 w-(--titlebar-controls-left,14px) [-webkit-app-region:drag]" />
        <div className="pointer-events-none absolute inset-y-0 left-[calc(var(--titlebar-controls-left,14px)+(var(--titlebar-control-size,24px)*2)+0.75rem)] right-[calc(var(--titlebar-tools-right,0.75rem)+0.75rem)] [-webkit-app-region:drag]" />
      </div>
      {state && !state.closed && (
        <PaneTabStrip
          listRef={strip}
          trailing={
            <PaneStripGlyph
              icon={<Codicon name="add" />}
              label={t.preview.newBrowserTab}
              onSelect={() => runBrowserTabAction('session.newTab')}
            />
          }
        >
          {state.tabs.map(tab => {
            const label = pages[tab.id]?.title || tab.target.label

            return (
              <PaneTab
                active={tab.id === activeId}
                data-tree-tab={tab.id}
                key={tab.id}
                onClose={() => void commandBrowserWorkspace({ kind: 'close', tabId: tab.id })}
              >
                <Tip label={label}>
                  <PaneTabLabel
                    aria-controls={`browser-page-${tab.id}`}
                    aria-selected={tab.id === activeId}
                    as="button"
                    className="normal-case tracking-normal"
                    onClick={() => void commandBrowserWorkspace({ kind: 'select', tabId: tab.id })}
                    role="tab"
                  >
                    {label}
                  </PaneTabLabel>
                </Tip>
              </PaneTab>
            )
          })}
        </PaneTabStrip>
      )}
      <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
        {state?.tabs.map(tab => {
          const active = tab.id === activeId

          return (
            <div
              id={`browser-page-${tab.id}`}
              key={tab.id}
              role="tabpanel"
              {...hiddenPaneProps(!active)}
              aria-hidden={!active}
              className="absolute inset-0"
              data-browser-active={active ? '' : undefined}
              inert={!active}
              style={{ visibility: active ? 'visible' : 'hidden' }}
            >
              <PaneVisibleContext.Provider value={active}>
                <PreviewTilePane
                  onClose={() => void commandBrowserWorkspace({ kind: 'close', tabId: tab.id })}
                  tabId={tab.id}
                />
              </PaneVisibleContext.Provider>
            </div>
          )
        })}
        {legacyTabId && <PreviewTilePane tabId={legacyTabId} />}
        {!state && !legacyTabId && (
          <div className="grid h-full place-items-center">
            <PanelEmpty description={t.preview.web.blankPageBody} icon="globe" />
          </div>
        )}
      </div>
    </div>
  )
}
