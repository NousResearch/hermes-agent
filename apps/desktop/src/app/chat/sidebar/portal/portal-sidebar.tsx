import './portal-sidebar.css'

import { useStore } from '@nanostores/react'
import { type ComponentProps, type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from 'react'

import { $sessionTileDragging } from '@/components/pane-shell/tree/store'
import { Sidebar, SidebarContent } from '@/components/ui/sidebar'
import { useI18n } from '@/i18n'
import {
  $portalAgentChatOrder,
  $portalAgentShowAll,
  agentSidebarKey,
  setAgentChatOrder,
  setAgentShowAll
} from '@/store/agent-sidebar-preferences'
import { $activeConnectionId } from '@/store/connections'
import { setPinnedSessionOrder } from '@/store/layout'
import {
  $pendingPortalProjectForNewChat,
  $portalAgentProjects,
  agentPortalProjectsFor,
  claimPendingPortalProjectSession,
  clearPendingPortalProjectNewChat,
  markPendingPortalProjectLeftArmed,
  mergeBackendPortalProjects,
  resolvePortalProjectForSession
} from '@/store/portal-projects'
import { syncPortalAgentProjects } from '@/store/portal-projects-sync'
import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import { $projects, $projectTree, refreshProjectTree } from '@/store/projects'
import { $gatewayState, $selectedStoredSessionId, sessionPinId } from '@/store/session'

import type { ChatSidebar } from '../index'
import { SidebarLoadMoreRow } from '../load-more-row'
import { ProjectDialog } from '../project-dialog'
import { WorktreeDialog } from '../projects/worktree-dialog'
import { $sessionListPinDragging } from '../reorderable-list'
import { SidebarLoadErrorState, SidebarSessionSkeletons, SidebarStorageCorruptNotice } from '../section-states'
import { SimplifiedAgentList } from '../simplified-agent-list'
import { useBotModeLoaded } from '../simplified-bot-chat-row'
import { SimplifiedNewSession } from '../simplified-new-session'
import { SimplifiedPinnedSessions } from '../simplified-pinned-sessions'
import { SimplifiedSessionList } from '../simplified-session-list'
import { useAgentSidebar } from '../use-agent-sidebar'
import { agentKey, agentOwner } from '../use-simplified-agents'

import { AGENT_CHAT_CAP, computeAgentChatLimits } from './agent-chat-cap'
import { arrangeAgentChats, orderAgentChats } from './agent-chat-order'
import { CreateAccountCard } from './create-account-card'
import { PortalAgentProjects } from './portal-agent-projects'
import { PortalCreateProjectDialog } from './portal-create-project-dialog'
import {
  type GroupChatMatches,
  PortalGroupChats,
  PortalNewGroupChatAction,
  useGroupChatMatches
} from './portal-group-chats'
import { PortalPinnedEmpty } from './portal-pinned-empty'
import { PortalSectionHeader } from './portal-section-header'
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

/** An empty Pinned (HADESK-132): the section stands down, but its drop target stays mounted — out
 *  of flow, so it costs no space — and comes out only while a session drag is in flight, so a chat
 *  dragged from the sidebar, or a session's own tab, can still be dropped onto Pinned (HADESK-58).
 *  Its box stands the same size with and without the caption and hint, so the drop geometry the
 *  drag snapshots at lift never moves under it. */
function PortalPinnedShelf() {
  const { t } = useI18n()
  // A session dragged by its own tab, or a sidebar row dragged by dnd-kit (whose list keeps no
  // dragging state the shelf could read), both bring the drop target out (HADESK-58, HADESK-132).
  const tileDragging = useStore($sessionTileDragging)
  const listDragging = useStore($sessionListPinDragging)
  const dragging = tileDragging || listDragging

  return (
    <section
      className="portal-section portal-pinned-shelf"
      data-dragging={dragging ? '' : undefined}
      data-session-pin-drop=""
      data-slot="pinned-shelf"
    >
      {dragging && (
        <>
          <PortalSectionHeader label={t.sidebar.pinned} />
          <PortalPinnedEmpty />
        </>
      )}
    </section>
  )
}

/** Sync backend `projects.db` records into `$portalAgentProjects` for every visible agent and
 *  mirror live `$projects` updates on the active profile so Classic and Redesign stay identical. */
function useSyncPortalBackendProjects(
  gatewayState: string,
  agents: readonly { connectionId: null | string; profile: string; targetProfile?: string }[],
  activeConnection: null | string,
  activeProfile: string
) {
  const activeBackendProjects = useStore($projects)
  const agentSyncKeys = agents.map(agent => agentSidebarKey(agentOwner(agent))).join('|')

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    const seen = new Set<string>()

    for (const agent of agents) {
      const owner = agentOwner(agent)
      const key = agentSidebarKey(owner)

      if (!seen.has(key)) {
        seen.add(key)
        void syncPortalAgentProjects(owner)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `agentSyncKeys` tracks the exact agent identity set.
  }, [agentSyncKeys, gatewayState])

  useEffect(() => {
    if (gatewayState !== 'open' || activeBackendProjects.length === 0) {
      return
    }

    mergeBackendPortalProjects(
      { connectionId: activeConnection ?? null, profile: activeProfile || 'default' },
      activeBackendProjects
    )
  }, [activeBackendProjects, activeConnection, activeProfile, gatewayState])
}

export function PortalSidebar(props: ComponentProps<typeof ChatSidebar>) {
  const gatewayState = useStore($gatewayState)

  // Warm registered Projects once connected so the composer project picker, Cmd+K,
  // and Move-to-project menu show real projects on first click without blocking boot.
  useEffect(() => {
    if (gatewayState !== 'open' || $projectTree.get().length > 0) {
      return
    }

    const timer = window.setTimeout(() => void refreshProjectTree(), 1_200)

    return () => window.clearTimeout(timer)
  }, [gatewayState])
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
  const sectionsRef = useRef<HTMLDivElement | null>(null)
  const [sectionsHeight, setSectionsHeight] = useState(0)
  const chatOrder = useStore($portalAgentChatOrder)
  const showAll = useStore($portalAgentShowAll)
  const allProjects = useStore($portalAgentProjects)
  const pendingProjectNewChat = useStore($pendingPortalProjectForNewChat)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const activeConnection = useStore($activeConnectionId)
  const activeProfile = useStore($activeGatewayProfile)
  useSyncPortalBackendProjects(gatewayState, agents.agents, activeConnection, activeProfile)

  // When a user clicked "+" / "New chat" inside a project, file the first newly created session
  // for that agent directly into that project. Never disarm on the pre-switch render while
  // `selectedStoredSessionId` still holds the chat we just navigated away from.
  useEffect(() => {
    if (!pendingProjectNewChat) {
      return
    }

    if (!selectedStoredSessionId) {
      markPendingPortalProjectLeftArmed()
    } else if (
      selectedStoredSessionId !== pendingProjectNewChat.armedAtSessionId ||
      pendingProjectNewChat.leftArmedSession
    ) {
      if (pendingProjectNewChat.baselineIds.has(selectedStoredSessionId)) {
        clearPendingPortalProjectNewChat()

        return
      }

      if (claimPendingPortalProjectSession(pendingProjectNewChat.owner, selectedStoredSessionId)) {
        return
      }
    }

    const ownerKey = agentSidebarKey(pendingProjectNewChat.owner)

    for (const agent of agents.agents) {
      if (agentSidebarKey(agentOwner(agent)) !== ownerKey) {
        continue
      }

      const list = agents.sessionsByAgent.get(agentKey(agent)) ?? []
      const fresh = list.find(row => row.id && !pendingProjectNewChat.baselineIds.has(row.id))

      if (fresh) {
        claimPendingPortalProjectSession(pendingProjectNewChat.owner, fresh.id)
        break
      }
    }
  }, [agents, pendingProjectNewChat, selectedStoredSessionId])
  useEffect(() => {
    const el = sectionsRef.current

    if (!el || typeof ResizeObserver === 'undefined') {
      return
    }

    const update = () => {
      const next = Math.round(el.clientHeight)
      setSectionsHeight(prev => (Math.abs(prev - next) >= 12 ? next : prev))
    }

    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)

    return () => observer.disconnect()
  }, [])
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

    // An empty Group chats leaves no section behind: New group chat moves up to the Agents header
    // so the first one is still a click away (HADESK-132).
    const noGroupChats = !searching && groups && groups.rooms.length + groups.names.length === 0

    const machineIds = new Set(agents.agents.map(a => a.connectionId))
    const openAgents: { key: string; count: number; showAll: boolean; active: boolean }[] = []
    let collapsedAgentCount = 0

    let projectRowsOverhead = 0

    for (const agent of agents.agents) {
      const isOpen = expanded[agentKey(agent)] ?? true
      const owner = agentOwner(agent)
      const key = agentSidebarKey(owner)

      if (!isOpen) {
        collapsedAgentCount += 1
        continue
      }

      const agentProjects = agentPortalProjectsFor(allProjects, owner)
      const agentSessions = agents.sessionsByAgent.get(agentKey(agent)) ?? []
      const looseCount = agentSessions.filter(row => !resolvePortalProjectForSession(agentProjects, row)).length
      projectRowsOverhead += agentProjects.projects.length

      openAgents.push({
        key,
        count: looseCount,
        showAll: showAll[key] ?? false,
        active: agent.connectionId === activeConnection && agent.profile === activeProfile
      })
    }

    const dynamicLimits = computeAgentChatLimits({
      availableHeight: Math.max(0, sectionsHeight - projectRowsOverhead * 36),
      collapsedAgentCount,
      groupRows: groups ? groups.rooms.length + groups.names.length : 0,
      machineCount: machineIds.size,
      openAgents,
      pinnedRows: pinned.length
    })
    return (
      <>
        {nothingFound && (
          <p className="portal-sidebar-empty" role="status">
            {t.sidebar.noMatch(query)}
          </p>
        )}
        <div className="portal-sidebar-sections" hidden={nothingFound} onKeyDown={moveRowFocus} ref={sectionsRef}>
          {pinned.length > 0 ? (
            <SimplifiedPinnedSessions
              {...shared}
              onReorder={rows => setPinnedSessionOrder(rows.map(row => sessionPinId(row)))}
              onResumeSession={resumePinned}
              sessions={pinned}
            />
          ) : (
            !searching && <PortalPinnedShelf />
          )}
          {groups && <PortalGroupChats matches={groups} />}
          <SimplifiedAgentList
            agents={agents}
            allSessions={all}
            expanded={expanded}
            headerAction={noGroupChats ? <PortalNewGroupChatAction /> : undefined}
            onResumeSession={shared.onResumeSession}
            onToggle={(id, open) => setExpanded(current => ({ ...current, [id]: !open }))}
            query={query}
            renderSessions={(sessions, agent) => {
              const owner = agentOwner(agent)
              const key = agentSidebarKey(owner)
              const agentShowsAll = showAll[key] ?? false
              const agentProjects = agentPortalProjectsFor(allProjects, owner)
              const ordered = orderAgentChats(sessions, chatOrder[key])
              const sessionsByProject = new Map<string, typeof ordered>()
              const looseSessions: typeof ordered = []

              for (const row of ordered) {
                const projectId = resolvePortalProjectForSession(agentProjects, row)

                if (projectId) {
                  const list = sessionsByProject.get(projectId) ?? []
                  list.push(row)
                  sessionsByProject.set(projectId, list)
                } else {
                  looseSessions.push(row)
                }
              }

              return (
                <div className="portal-agent-sessions">
                  <PortalAgentProjects
                    agent={agent}
                    allAgentSessionIds={ordered.map(row => row.id)}
                    owner={owner}
                    projects={agentProjects.projects}
                    renderChats={projectChats => (
                      <SimplifiedSessionList {...shared} searching={searching} sessions={projectChats} />
                    )}
                    searching={searching}
                    sessionsByProject={sessionsByProject}
                  />
                  <SimplifiedSessionList
                    {...shared}
                    cap={{
                      limit: dynamicLimits[key] ?? AGENT_CHAT_CAP,
                      showAll: agentShowsAll,
                      onToggle: () => setAgentShowAll(owner, !agentShowsAll)
                    }}
                    onReorder={rows => setAgentChatOrder(owner, arrangeAgentChats(rows))}
                    sessions={looseSessions}
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
        {/* One slot under the header: New chat's row, or the search field in its place while it is
            out — never both, so the sections below keep their place when search opens (HADESK-53).
            SimplifiedNewSession stays mounted (`hidden`) so its `session.new` agent claim survives
            while search is open (HADESK-78). */}
        {searchBox.open && (
          <PortalSidebarSearchField onChange={setQuery} pending={search.pending} query={query} search={searchBox} />
        )}
        <SimplifiedNewSession
          hidden={searchBox.open}
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
        <SidebarStorageCorruptNotice />
        {search.failed && (
          <p className="text-[length:var(--portal-type-meta-size)] text-destructive" role="alert">
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
      <ProjectDialog />
      <PortalCreateProjectDialog />
      {/* One mount for the whole app. The header of WorktreeDialog tells why. */}
      <WorktreeDialog />
      <CreateAccountCard />
      <UsageStatsCard />
    </Sidebar>
  )
}
