import { useStore } from '@nanostores/react'
import type { CSSProperties } from 'react'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

import { BrowserRegion, ChatRegion, EditorRegion, ExplorerRegion } from '../regions'
import { $ideEditor } from '../regions/editor/tabs'
import { $ideWorkspaceRoot, workspaceBasename } from '../state'

import {
  $ideLayout,
  setIdeBrowserHeight,
  setIdeChatWidth,
  setIdeExplorerWidth,
  toggleIdeBrowser,
  toggleIdeChat,
  toggleIdeExplorer
} from './ide-layout'
import { IdeSplitHandle } from './ide-split-handle'

/**
 * Dedicated shell for `?win=ide` — the Hermes IDE: a fixed four-region
 * workbench (explorer | editor + browser | chat) over the same gateway every
 * other window uses. The frame owns region geometry and the workspace readout;
 * each region owns its own content.
 */
export function IdeShell() {
  const { t } = useI18n()
  const layout = useStore($ideLayout)
  const workspaceRoot = useStore($ideWorkspaceRoot)
  const editor = useStore($ideEditor)

  return (
    <div
      className="flex h-screen min-h-0 w-screen flex-col overflow-hidden bg-(--ui-bg-chrome) text-(--ui-text-primary)"
      data-contrib-shell=""
      style={{ '--titlebar-height': `${TITLEBAR_HEIGHT}px` } as CSSProperties}
    >
      <header
        className="relative flex shrink-0 items-center border-b border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)"
        style={{ height: TITLEBAR_HEIGHT }}
      >
        {/* Native-window drag: the strip runs between the reserved traffic-light
            and window-controls insets, exactly like the app titlebar and the
            browser pop-out — a full-bar drag region would eat the buttons. */}
        <div className="pointer-events-none absolute inset-y-0 left-0 w-(--titlebar-controls-left,14px) [-webkit-app-region:drag]" />
        <div className="pointer-events-none absolute inset-y-0 left-[calc(var(--titlebar-controls-left,14px)+(var(--titlebar-control-size,24px)*2)+0.75rem)] right-[calc(var(--titlebar-tools-right,0.75rem)+0.75rem)] [-webkit-app-region:drag]" />
        <div className="relative z-1 flex min-w-0 items-center gap-2 pl-[max(0.75rem,var(--titlebar-controls-left,0px))] pr-[calc(var(--titlebar-tools-right,0.75rem)+0.75rem)]">
          <span className="flex shrink-0 items-center gap-1.5 text-[12px] font-medium text-(--ui-text-secondary)">
            <Codicon name="code" size={14} />
            {t.ide.title}
          </span>
          <span className="min-w-0 truncate text-[12px] text-(--ui-text-tertiary)">
            {workspaceBasename(workspaceRoot) ?? t.ide.noWorkspace}
          </span>
        </div>
      </header>

      <div className="flex min-h-0 min-w-0 flex-1">
        {layout.explorerOpen && (
          <>
            <div className="min-h-0 shrink-0" style={{ width: layout.explorerWidth }}>
              <ExplorerRegion />
            </div>
            <IdeSplitHandle
              axis="x"
              label={t.ide.resizeExplorer}
              setSize={setIdeExplorerWidth}
              size={() => $ideLayout.get().explorerWidth}
            />
          </>
        )}

        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {/* The editor column exists only while a file is open: an empty
              frame is dead weight above the browser, and closing the last tab
              (or never opening one) should give its space back. */}
          {editor.openPaths.length > 0 && (
            <div className="flex min-h-0 min-w-0 flex-1">
              <EditorRegion />
            </div>
          )}

          {layout.browserOpen && (
            <>
              <IdeSplitHandle
                axis="y"
                invert
                label={t.ide.resizeBrowser}
                setSize={setIdeBrowserHeight}
                size={() => $ideLayout.get().browserHeight}
              />
              <div className="min-h-0 shrink-0" style={{ height: layout.browserHeight }}>
                <BrowserRegion />
              </div>
            </>
          )}
        </div>

        {layout.chatOpen && (
          <>
            <IdeSplitHandle
              axis="x"
              invert
              label={t.ide.resizeChat}
              setSize={setIdeChatWidth}
              size={() => $ideLayout.get().chatWidth}
            />
            <div className="min-h-0 shrink-0" style={{ width: layout.chatWidth }}>
              <ChatRegion />
            </div>
          </>
        )}
      </div>

      <footer className="flex h-6 shrink-0 items-center justify-between gap-2 border-t border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome) px-2 text-[11px] text-(--ui-text-tertiary)">
        <span className="min-w-0 truncate pl-1">{workspaceRoot ?? t.ide.noWorkspace}</span>
        <span className="flex shrink-0 items-center gap-0.5">
          <Tip label={t.ide.toggleExplorer}>
            <Button
              aria-label={t.ide.toggleExplorer}
              aria-pressed={layout.explorerOpen}
              onClick={toggleIdeExplorer}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="layout-sidebar-left" size={13} />
            </Button>
          </Tip>
          <Tip label={t.ide.toggleChat}>
            <Button
              aria-label={t.ide.toggleChat}
              aria-pressed={layout.chatOpen}
              onClick={toggleIdeChat}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="comment-discussion" size={13} />
            </Button>
          </Tip>
          <Tip label={t.ide.toggleBrowser}>
            <Button
              aria-label={t.ide.toggleBrowser}
              aria-pressed={layout.browserOpen}
              onClick={toggleIdeBrowser}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="globe" size={13} />
            </Button>
          </Tip>
        </span>
      </footer>
    </div>
  )
}
