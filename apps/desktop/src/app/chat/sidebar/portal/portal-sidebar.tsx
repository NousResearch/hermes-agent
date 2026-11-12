import './portal-sidebar.css'

import { type ComponentProps, type KeyboardEvent, useEffect, useRef } from 'react'

import wing from '@/assets/catalog-wing.svg'
import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { Sidebar, SidebarContent } from '@/components/ui/sidebar'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ArrowBarToLeft as IconArrowBarToLeft } from '@/lib/icons'
import { agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { recordIaAction } from '@/store/ia-metrics'
import { SESSION_SEARCH_FOCUS_EVENT, toggleSidebarOpen } from '@/store/layout'
import { $newChatProfile } from '@/store/profile'

import type { ChatSidebar } from '../index'
import { SidebarLoadMoreRow } from '../load-more-row'
import { SidebarLoadErrorState, SidebarSessionSkeletons, SidebarStorageCorruptNotice } from '../section-states'
import { SimplifiedAgentList } from '../simplified-agent-list'
import { SimplifiedNewSession } from '../simplified-new-session'
import { SimplifiedPinnedSessions } from '../simplified-pinned-sessions'
import { SimplifiedSessionList } from '../simplified-session-list'
import { useAgentSidebar } from '../use-agent-sidebar'

import { AGENT_CHAT_CAP, agentShowAllKey } from './agent-chat-cap'
import { PortalGroupChats } from './portal-group-chats'
import { PortalPinnedEmpty } from './portal-pinned-empty'

const ROW_STEP: Record<string, (index: number, count: number) => number> = {
  ArrowDown: (index, count) => Math.min(index + 1, count - 1),
  ArrowUp: index => Math.max(index - 1, 0),
  Home: () => 0,
  End: (_, count) => count - 1
}

/** Arrow keys walk the visible rows (sections, machines, agents, chats); Tab still reaches row actions. */
function moveRowFocus(event: KeyboardEvent<HTMLElement>) {
  const step = ROW_STEP[event.key]
  const target = event.target as HTMLElement

  if (!step || target.dataset.slot !== 'row-button') {
    return
  }

  const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('[data-slot="row-button"]')].filter(
    row => !row.closest('[hidden]')
  )

  const next = rows[step(rows.indexOf(target), rows.length)]

  if (next && next !== target) {
    event.preventDefault()
    next.focus()
  }
}

interface PortalSidebarListStatesProps {
  empty: boolean
  error: boolean
  hasMore: boolean
  loadMore: () => void
  loading: boolean
  loadingMore: boolean
  onRetry: () => void
  pendingSearch: boolean
  showSkeletons: boolean
}

/** What sits under the sections: skeletons while loading, the no-sessions note, paging and errors. */
function PortalSidebarListStates(props: PortalSidebarListStatesProps) {
  const { t } = useI18n()

  return (
    <>
      {props.showSkeletons && <SidebarSessionSkeletons />}
      {props.pendingSearch && (
        <p className="sr-only" role="status">
          {t.common.loading}
        </p>
      )}
      {props.empty && <p className="portal-sidebar-empty">{t.sidebar.noSessions}</p>}
      {props.hasMore && (
        <SidebarLoadMoreRow loading={props.loading || props.loadingMore} onClick={props.loadMore} step={0} />
      )}
      {props.error && <SidebarLoadErrorState onRetry={props.onRetry} />}
    </>
  )
}

export function PortalSidebar(props: ComponentProps<typeof ChatSidebar>) {
  const {
    t,
    all,
    query,
    setQuery,
    search,
    rows,
    shared,
    agents,
    expanded,
    setExpanded,
    revealSession,
    loading,
    error,
    loadingMore,
    loadMore,
    resumePinned,
    pinned,
    hasMore
  } = useAgentSidebar(props)

  const searchInputRef = useRef<HTMLInputElement>(null)
  const searching = Boolean(query.trim())
  // Nothing in flight: a settled list may show its empty notes.
  const quiet = !loading && !search.pending && !search.failed
  const settled = quiet && rows.length === 0

  useEffect(() => {
    const onFocus = () => searchInputRef.current?.focus({ preventScroll: true })
    window.addEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)

    return () => window.removeEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)
  }, [])

  return (
    <Sidebar className="portal-sidebar" collapsible="none" data-tour="sessions-sidebar">
      <header className="portal-sidebar-brand">
        <span aria-hidden className="portal-sidebar-badge">
          <img alt="" src={wing} />
        </span>
        <div className="portal-sidebar-identity">
          <span className="portal-sidebar-wordmark">Hermes</span>
          <span className="portal-sidebar-byline">// Nous Research</span>
        </div>
        <Tip label={t.titlebar.hideSidebar}>
          <Button
            aria-label={t.titlebar.hideSidebar}
            className="portal-sidebar-collapse"
            onClick={toggleSidebarOpen}
            size="icon-xs"
            variant="ghost"
          >
            <IconArrowBarToLeft aria-hidden />
          </Button>
        </Tip>
      </header>
      <SidebarContent className="portal-sidebar-content">
        <SimplifiedNewSession
          onFallback={() => {
            $newChatProfile.set(null)
            props.onNavigate({
              id: 'new-session',
              label: t.sidebar.nav['new-session'],
              action: 'new-session',
              icon: () => null,
              keybindActionId: 'session.new'
            })
          }}
          sessions={all}
        />
        <SearchField
          aria-label={t.sidebar.searchAria}
          containerClassName="portal-sidebar-search"
          inputRef={searchInputRef}
          loading={search.pending}
          onChange={setQuery}
          onFocus={() => recordIaAction('ia.search.opened')}
          placeholder={t.sidebar.searchPlaceholder}
          value={query}
        />
        <SidebarStorageCorruptNotice />
        {search.failed && (
          <p className="text-xs text-destructive" role="alert">
            {t.errors.genericFailure}
          </p>
        )}
        {/* A search with no hits answers right under the field; the empty sections stand down. */}
        {searching && settled && (
          <p className="portal-sidebar-empty" role="status">
            {t.sidebar.noMatch(query)}
          </p>
        )}
        <div className="portal-sidebar-sections" hidden={searching && settled} onKeyDown={moveRowFocus}>
          {(!searching || pinned.length > 0) && (
            <SimplifiedPinnedSessions
              {...shared}
              emptyState={!searching && quiet && !error && <PortalPinnedEmpty />}
              onResumeSession={resumePinned}
              sessions={pinned}
            />
          )}
          <PortalGroupChats query={query} />
          <SimplifiedAgentList
            agents={agents}
            allSessions={all}
            expanded={expanded}
            onResumeSession={shared.onResumeSession}
            onToggle={(id, open) => setExpanded(current => ({ ...current, [id]: !open }))}
            query={query}
            renderSessions={(sessions, agent) => {
              const showAllKey = agentShowAllKey(
                agentSidebarKey({ connectionId: agent.connectionId, profile: agent.targetProfile || agent.profile })
              )

              const showAll = expanded[showAllKey] ?? false

              return (
                <div className="portal-agent-sessions">
                  <SimplifiedSessionList
                    {...shared}
                    cap={{
                      limit: AGENT_CHAT_CAP,
                      showAll,
                      onToggle: () => setExpanded(current => ({ ...current, [showAllKey]: !showAll }))
                    }}
                    sessions={sessions}
                  />
                </div>
              )
            }}
            revealSession={revealSession}
          />
        </div>
        <PortalSidebarListStates
          empty={!searching && settled}
          error={Boolean(error)}
          hasMore={hasMore}
          loading={loading}
          loadingMore={loadingMore}
          loadMore={() => void loadMore()}
          onRetry={() => void props.onRetrySessions()}
          pendingSearch={searching && search.pending}
          showSkeletons={loading && rows.length === 0}
        />
      </SidebarContent>
    </Sidebar>
  )
}
