import { useStore } from '@nanostores/react'
import { useCallback, useRef, useState } from 'react'

import { requestComposerInsert, requestComposerInsertRefs } from '@/app/chat/composer/focus'
import { PreviewTilePane } from '@/app/chat/right-rail/preview'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $rightRailActiveTabId, selectRightRailTab } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { $previewTabs, closeRightRailTab, newBrowserTab, openBrowserTab } from '@/store/preview'

import { $ideActiveChat } from '../chat/store'

import { parsePickedPayload, pickedElementRef, PICKER_SCRIPT } from './inspect'

interface WebviewLike extends HTMLElement {
  executeJavaScript?: (code: string) => Promise<unknown>
}

/**
 * Browser pane: the shared in-app browser (same webview, partition, address bar
 * and console the preview rail uses) hosted as the IDE's own tab strip. The
 * agent can drive this page (`drive_preview` targets the session's client —
 * this window), and the user can push what they see into the chat: Inspect
 * picks an element (selector + outerHTML) and Add page sends the current URL.
 */
export function BrowserRegion() {
  const { t } = useI18n()
  const tabs = useStore($previewTabs)
  const activeTabId = useStore($rightRailActiveTabId)
  const active = tabs.find(tab => tab.id === activeTabId) ?? tabs[tabs.length - 1] ?? null
  const [picking, setPicking] = useState(false)
  const regionRef = useRef<HTMLDivElement | null>(null)

  // The IDE chat column's active composer (or the ambient one when no IDE
  // session is open — the insert simply finds no target then).
  const chatTarget = () => {
    const activeChat = $ideActiveChat.get()

    return activeChat ? `tile:${activeChat}` : 'active'
  }

  const addToChat = useCallback((text: string) => {
    requestComposerInsert(text, { mode: 'block', target: chatTarget() })
  }, [])

  const inspectElement = useCallback(async () => {
    const webview = regionRef.current?.querySelector('webview') as null | WebviewLike

    if (!webview?.executeJavaScript) {
      notifyError(new Error('no webview in the IDE browser pane'), t.ide.browserInspectFailed)

      return
    }

    setPicking(true)

    try {
      const picked = parsePickedPayload(await webview.executeJavaScript(PICKER_SCRIPT))

      if (picked) {
        // The pick travels as a collapsed `@element:` chip (selector visible,
        // HTML in the payload) rather than a code block dumped into the
        // composer — same shape Cursor/VS Code attach a picked element in.
        requestComposerInsertRefs([pickedElementRef(picked, active?.target.url ?? '')], { target: chatTarget() })
      }
    } catch (error) {
      notifyError(error, t.ide.browserInspectFailed)
    } finally {
      setPicking(false)
    }
  }, [active?.target.url, addToChat, t.ide.browserInspectFailed])

  return (
    <section
      aria-label={t.ide.browserTitle}
      className="flex h-full min-h-0 w-full min-w-0 flex-col bg-(--ui-chat-surface-background)"
    >
      <header
        className="flex h-9 shrink-0 items-stretch gap-0.5 border-b border-(--ui-stroke-tertiary) px-1 pt-1"
      >
        {/* Tabs scroll inside their own region; the strip's controls sit right
            after them and stay visible while the tabs scroll. */}
        <div
          aria-label={t.ide.browserTabsLabel}
          className="flex min-w-0 shrink items-stretch gap-0.5 overflow-x-auto"
          role="tablist"
        >
          {tabs.map(tab => {
          const isActive = tab.id === active?.id

          return (
            <div
              aria-selected={isActive}
              className={cn(
                'group flex h-8 max-w-56 min-w-0 shrink-0 items-center gap-1.5 rounded-t-sm border border-b-0 border-transparent px-2 text-xs',
                isActive
                  ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) text-foreground'
                  : 'text-(--ui-text-secondary) hover:bg-(--ui-bg-quaternary)'
              )}
              key={tab.id}
              role="tab"
            >
              <button
                className="flex min-w-0 items-center gap-1.5"
                onClick={() => selectRightRailTab(tab.id)}
                type="button"
              >
                <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="globe" size={13} />
                <span className="truncate">{tab.target.label}</span>
              </button>
              <Tip label={t.ide.closeTab}>
                <button
                  aria-label={t.ide.closeTabLabel(tab.target.label)}
                  className="grid size-4 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 group-hover:opacity-100 hover:bg-(--ui-bg-quaternary) hover:text-foreground"
                  onClick={() => closeRightRailTab(tab.id)}
                  type="button"
                >
                  <Codicon name="close" size={12} />
                </button>
              </Tip>
            </div>
          )
        })}
        </div>
        <span className="my-auto ml-1 flex shrink-0 items-center gap-0.5">
        <Tip label={t.ide.browserNewTab}>
          <Button
            aria-label={t.ide.browserNewTab}
            className="my-auto"
            onClick={() => newBrowserTab()}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="add" size={13} />
          </Button>
        </Tip>
        {active && (
          <>
            <Tip label={t.ide.browserInspect}>
              <Button
                aria-label={t.ide.browserInspect}
                className="my-auto"
                disabled={picking}
                onClick={() => void inspectElement()}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Codicon name="inspect" size={13} />
              </Button>
            </Tip>
            <Tip label={t.ide.browserAddPage}>
              <Button
                aria-label={t.ide.browserAddPage}
                className="my-auto"
                onClick={() => addToChat(`Page: ${active.target.url}`)}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Codicon name="link" size={13} />
              </Button>
            </Tip>
          </>
        )}
        </span>
      </header>
      <div className="relative min-h-0 flex-1 overflow-hidden" ref={regionRef}>
        {active ? (
          <PreviewTilePane tabId={active.id} />
        ) : (
          <div className="flex h-full flex-col">
            <EmptyState className="min-h-0 pt-8" description={t.ide.browserEmptyBody} title={t.ide.browserEmptyTitle} />
            <div className="flex shrink-0 justify-center pb-3">
              <Button onClick={() => openBrowserTab()} size="sm" variant="secondary">
                {t.ide.browserOpen}
              </Button>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
