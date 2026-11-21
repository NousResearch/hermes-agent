import './portal-sidebar.css'

import { useStore } from '@nanostores/react'
import { type ComponentProps, type KeyboardEvent, type ReactNode } from 'react'

import { Sidebar, SidebarContent } from '@/components/ui/sidebar'
import { useI18n } from '@/i18n'
import {
  $portalAgentChatOrder,
  $portalAgentShowAll,
  agentSidebarKey,
  setAgentChatOrder,
  setAgentShowAll
} from '@/store/agent-sidebar-preferences'
import { setPinnedSessionOrder } from '@/store/layout'
import { $newChatProfile } from '@/store/profile'
import { sessionPinId } from '@/store/session'

import type { ChatSidebar } from '../index'
import { SidebarLoadMoreRow } from '../load-more-row'
import { SidebarLoadErrorState, SidebarSessionSkeletons, SidebarStorageCorruptNotice } from '../section-states'
import { SimplifiedAgentList } from '../simplified-agent-list'
import { useBotModeLoaded } from '../simplified-bot-chat-row'
import { SimplifiedNewSession } from '../simplified-new-session'
import { SimplifiedPinnedSessions } from '../simplified-pinned-sessions'
import { SimplifiedSessionList } from '../simplified-session-list'
import { useAgentSidebar } from '../use-agent-sidebar'
import { agentOwner } from '../use-simplified-agents'

import { AGENT_CHAT_CAP } from './agent-chat-cap'
import { arrangeAgentChats, orderAgentChats } from './agent-chat-order'
import { CreateAccountCard } from './create-account-card'
import { type GroupChatMatches, PortalGroupChats, useGroupChatMatches } from './portal-group-chats'
import { PortalPinnedEmpty } from './portal-pinned-empty'
import { PortalSidebarHead, PortalSidebarSearchField, usePortalSidebarSearch } from './portal-sidebar-head'
import { UsageStatsCard } from './usage-stats-card'

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

/** Group chats are Bot Mode's: their search runs only inside this, mounted while the plugin is loaded. */
function BotModeGroupChats({ children, query }: { children: (groups: GroupChatMatches) => ReactNode; query: string }) {
  return children(useGroupChatMatches(query))
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

  const searchBox = usePortalSidebarSearch(setQuery)
  const chatOrder = useStore($portalAgentChatOrder)
  const showAll = useStore($portalAgentShowAll)
  // Group chats are Bot Mode's: a disabled or failed plugin leaves no section behind.
  const botModeLoaded = useBotModeLoaded()
  const searching = Boolean(query.trim())
  // Nothing in flight: a settled list may show its empty notes.
  const quiet = !loading && !search.pending && !search.failed
  const settled = quiet && rows.length === 0
  const needle = query.trim().toLowerCase()

  // A search answers from every section: chats (pinned or under an agent), agents by name, and
  // group chats; only when none of them holds a match does it say so.
  const sections = (groups?: GroupChatMatches) => {
    const nothingFound =
      searching &&
      settled &&
      pinned.length === 0 &&
      !agents.agents.some(agent => agent.label.toLowerCase().includes(needle)) &&
      !(groups && groups.rooms.length + groups.names.length > 0)

    return (
      <>
        {nothingFound && (
          <p className="portal-sidebar-empty" role="status">
            {t.sidebar.noMatch(query)}
          </p>
        )}
        <div className="portal-sidebar-sections" hidden={nothingFound} onKeyDown={moveRowFocus}>
          {(!searching || pinned.length > 0) && (
            <SimplifiedPinnedSessions
              {...shared}
              emptyState={!searching && quiet && !error && <PortalPinnedEmpty />}
              onReorder={rows => setPinnedSessionOrder(rows.map(row => sessionPinId(row)))}
              onResumeSession={resumePinned}
              sessions={pinned}
            />
          )}
          {groups && <PortalGroupChats matches={groups} searching={searching} />}
          <SimplifiedAgentList
            agents={agents}
            allSessions={all}
            expanded={expanded}
            onResumeSession={shared.onResumeSession}
            onToggle={(id, open) => setExpanded(current => ({ ...current, [id]: !open }))}
            query={query}
            renderSessions={(sessions, agent) => {
              const owner = agentOwner(agent)
              const key = agentSidebarKey(owner)
              const agentShowsAll = showAll[key] ?? false

              return (
                <div className="portal-agent-sessions">
                  <SimplifiedSessionList
                    {...shared}
                    cap={{
                      limit: AGENT_CHAT_CAP,
                      showAll: agentShowsAll,
                      onToggle: () => setAgentShowAll(owner, !agentShowsAll)
                    }}
                    onReorder={rows => setAgentChatOrder(owner, arrangeAgentChats(rows))}
                    sessions={orderAgentChats(sessions, chatOrder[key])}
                  />
                </div>
              )
            }}
            revealSession={revealSession}
          />
        </div>
      </>
    )
  }

  return (
    <Sidebar className="portal-sidebar" collapsible="none" data-tour="sessions-sidebar">
      <SidebarContent className="portal-sidebar-content">
        <PortalSidebarHead search={searchBox} />
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
        {searchBox.open && (
          <PortalSidebarSearchField onChange={setQuery} pending={search.pending} query={query} search={searchBox} />
        )}
        <SidebarStorageCorruptNotice />
        {search.failed && (
          <p className="text-xs text-destructive" role="alert">
            {t.errors.genericFailure}
          </p>
        )}
        {/* A search with no hits answers right under the field; the empty sections stand down. */}
        {botModeLoaded ? <BotModeGroupChats query={query}>{sections}</BotModeGroupChats> : sections()}
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
      <CreateAccountCard />
      <UsageStatsCard />
    </Sidebar>
  )
}
