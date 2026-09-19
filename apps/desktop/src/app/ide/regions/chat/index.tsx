import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { requestCloseSessionTile, SessionTilePane, tileStoredRow } from '@/app/chat/session-tile'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { Loader } from '@/components/ui/loader'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { NEW_SESSION_TITLE, sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $gatewayState } from '@/store/session'
import { $sessionTiles, openSessionTile } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { createIdeSession, listIdeSessions, rememberIdeSessionRows } from './sessions'
import { $ideActiveChat, activateIdeChat } from './store'

/**
 * IDE chat column: the window's own session tabs (source='ide'), each mounting
 * the REAL chat surface — the same transcript, composer, and tool activity the
 * main window renders, resumed by the shared session-tile machinery. The tabs
 * live only in this window and never appear in the primary sidebar.
 */
export function ChatRegion() {
  const { t } = useI18n()
  const { requestGateway } = useGatewayRequest()
  const tabs = useStore($sessionTiles)
  const active = useStore($ideActiveChat)
  const gatewayState = useStore($gatewayState)
  const [creating, setCreating] = useState(false)
  const [recent, setRecent] = useState<SessionInfo[]>([])

  const effective = tabs.find(tile => tile.storedSessionId === active)?.storedSessionId ?? tabs[0]?.storedSessionId ?? null

  // Keep the pointer on a living tab (its tab was closed, or a profile swap
  // replaced the set).
  useEffect(() => {
    if (active && !tabs.some(tile => tile.storedSessionId === active)) {
      activateIdeChat(tabs[0]?.storedSessionId ?? null)
    }
  }, [active, tabs])

  // Refresh the IDE session slice once the gateway can answer: feeds both the
  // reopen list and tab titles (ide rows never arrive via the primary lists).
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    let alive = true

    void listIdeSessions()
      .then(sessions => {
        if (alive) {
          setRecent(sessions)
          rememberIdeSessionRows(sessions)
        }
      })
      .catch(() => undefined)

    return () => {
      alive = false
    }
  }, [gatewayState])

  const startSession = useCallback(async () => {
    setCreating(true)

    try {
      const stored = await createIdeSession(requestGateway)

      if (!stored) {
        notifyError(new Error('session.create returned no stored id'), t.ide.chatCreateFailed)
      }
    } catch (error) {
      notifyError(error, t.ide.chatCreateFailed)
    } finally {
      setCreating(false)
    }
  }, [requestGateway, t.ide.chatCreateFailed])

  const openRecent = (sessionId: string) => {
    openSessionTile(sessionId)
    activateIdeChat(sessionId)
  }

  return (
    <section
      aria-label={t.ide.chatTitle}
      className="flex h-full min-h-0 w-full min-w-0 flex-col bg-(--ui-bg-chrome)"
    >
      <header
        aria-label={t.ide.chatTabsLabel}
        className="flex h-9 shrink-0 items-stretch gap-0.5 overflow-x-auto border-b border-(--ui-stroke-tertiary) px-1 pt-1"
        role="tablist"
      >
        {tabs.map(tile => {
          const row = tileStoredRow(tile.storedSessionId)
          const title = row ? sessionTitle(row) : NEW_SESSION_TITLE
          const isActive = tile.storedSessionId === effective

          return (
            <div
              aria-selected={isActive}
              className={cn(
                'group flex h-8 max-w-56 min-w-0 shrink-0 items-center gap-1.5 rounded-t-sm border border-b-0 border-transparent px-2 text-xs',
                isActive
                  ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) text-foreground'
                  : 'text-(--ui-text-secondary) hover:bg-(--ui-bg-quaternary)'
              )}
              key={tile.storedSessionId}
              role="tab"
            >
              <button
                className="flex min-w-0 items-center gap-1.5"
                onClick={() => activateIdeChat(tile.storedSessionId)}
                type="button"
              >
                <span className="truncate">{title}</span>
              </button>
              <Tip label={t.ide.closeTab}>
                <button
                  aria-label={t.ide.closeTabLabel(title)}
                  className="grid size-4 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 group-hover:opacity-100 hover:bg-(--ui-bg-quaternary) hover:text-foreground"
                  onClick={() => requestCloseSessionTile(tile.storedSessionId)}
                  type="button"
                >
                  <Codicon name="close" size={12} />
                </button>
              </Tip>
            </div>
          )
        })}
        <Tip label={t.ide.chatNew}>
          <Button
            aria-label={t.ide.chatNew}
            className="my-auto ml-1 shrink-0"
            disabled={creating}
            onClick={() => void startSession()}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            {creating ? <Loader /> : <Codicon name="add" size={13} />}
          </Button>
        </Tip>
      </header>

      <div className="min-h-0 flex-1 overflow-hidden">
        {effective ? (
          <SessionTilePane key={effective} storedSessionId={effective} />
        ) : (
          <div className="flex h-full min-h-0 flex-col">
            <EmptyState className="min-h-0 pt-10" description={t.ide.chatEmptyBody} title={t.ide.chatEmptyTitle} />
            <div className="flex shrink-0 justify-center pb-3">
              <Button disabled={creating} onClick={() => void startSession()} size="sm" variant="secondary">
                {t.ide.chatNew}
              </Button>
            </div>
            {recent.length > 0 && (
              <div className="min-h-0 flex-1 overflow-auto px-3 pb-3">
                <div className="mb-1 px-2 text-[11px] font-medium tracking-wider text-(--ui-text-tertiary) uppercase">
                  {t.ide.chatRecent}
                </div>
                {recent.map(session => (
                  <button
                    className="flex w-full items-center gap-2 rounded-sm px-2 py-1 text-left text-xs text-(--ui-text-secondary) hover:bg-(--ui-bg-quaternary) hover:text-foreground"
                    key={session.id}
                    onClick={() => openRecent(session.id)}
                    type="button"
                  >
                    <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="comment-discussion" size={13} />
                    <span className="truncate">{sessionTitle(session)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
