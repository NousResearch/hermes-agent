import './portal-sidebar.css'

import { useStore } from '@nanostores/react'
import { type ComponentProps, type KeyboardEvent, type ReactNode, useEffect, useState } from 'react'

import { $sessionTileDragging } from '@/components/pane-shell/tree/store'
import { Sidebar, SidebarContent } from '@/components/ui/sidebar'
import { useI18n } from '@/i18n'
import { $portalAgentChatOrder, agentSidebarKey, setAgentChatOrder } from '@/store/agent-sidebar-preferences'
import { $activeConnectionId } from '@/store/connections'
import { setPinnedSessionOrder } from '@/store/layout'
import {
  $pendingPortalProjectForNewChat,
  $portalAgentProjects,
  $portalProjectExpanded,
  agentPortalProjectsFor,
  claimPendingPortalProjectSession,
  clearPendingPortalProjectNewChat,
  markPendingPortalProjectLeftArmed,
  resolvePortalProjectForSession
} from '@/store/portal-projects'
import { syncPortalAgentProjects } from '@/store/portal-projects-sync'
import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import { $projects, $projectTree, refreshProjectTree } from '@/store/projects'
import { $gatewayState, $selectedStoredSessionId, knownSessionOwner, sessionPinId } from '@/store/session'

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

import { AGENT_CHAT_CAP, computeAgentChatLimits, computeShowMoreStep } from './agent-chat-cap'
import { arrangeAgentChats, orderAgentChats } from './agent-chat-order'
import { CreateAccountCard } from './create-account-card'
import { PortalAgentProjects } from './portal-agent-projects'
import { PortalBotAgentDialogs } from './portal-bot-agent-items'
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

function resolveAgentProjectSessions(
  agentProjects: ReturnType<typeof agentPortalProjectsFor>,
  agentSessions: readonly { id: string; cwd?: string | null }[]
): { looseCount: number; map: Map<string, string | null> } {
  const map = new Map<string, string | null>()
  let looseCount = 0

  for (const row of agentSessions) {
    const pid = resolvePortalProjectForSession(agentProjects, row)
    map.set(row.id, pid)

    if (!pid) {
      looseCount += 1
    }
  }

  return { looseCount, map }
}

function countExpandedProjectRows(
  projects: readonly { id: string }[],
  expandedProjects: Record<string, boolean>,
  resolvedMap: Map<string, string | null>
): number {
  let rows = projects.length

  for (const project of projects) {
    if (!expandedProjects[project.id]) {
      continue
    }

    let inProject = 0

    for (const pid of resolvedMap.values()) {
      if (pid === project.id) {
        inProject += 1
      }
    }

    rows += Math.max(1, inProject)
  }

  return rows
}

function buildAgentChatLimits(params: {
  agents: ReturnType<typeof useAgentSidebar>['agents']
  allProjects: ReturnType<typeof $portalAgentProjects.get>
  expanded: Record<string, boolean>
  expandedProjects: Record<string, boolean>
  activeConnection: string | null
  activeProfile: string | null
  sectionsHeight: number
  agentsViewportHeight: number
  screenChromePx: number
  groupRows: number
  pinnedRows: number
}) {
  const machineIds = new Set(params.agents.agents.map(a => a.connectionId))
  const openAgents: { key: string; count: number; showAll: boolean; active: boolean }[] = []
  const resolvedProjectsByAgent = new Map<string, Map<string, string | null>>()
  let collapsedAgentCount = 0
  let projectRowsOverhead = 0

  for (const agent of params.agents.agents) {
    const isOpen = params.expanded[agentKey(agent)] ?? true
    const owner = agentOwner(agent)
    const key = agentSidebarKey(owner)
    const agentProjects = agentPortalProjectsFor(params.allProjects, owner)
    const agentSessions = params.agents.sessionsByAgent.get(agentKey(agent)) ?? []
    const resolved = resolveAgentProjectSessions(agentProjects, agentSessions)
    resolvedProjectsByAgent.set(agentKey(agent), resolved.map)

    if (!isOpen) {
      collapsedAgentCount += 1
      continue
    }

    projectRowsOverhead += countExpandedProjectRows(agentProjects.projects, params.expandedProjects, resolved.map)
    openAgents.push({
      key,
      count: resolved.looseCount,
      showAll: false,
      active: agent.connectionId === params.activeConnection && agent.profile === params.activeProfile
    })
  }

  const projectPx = projectRowsOverhead * 40
  const dynamicLimits = computeAgentChatLimits({
    agentsViewportHeight: params.agentsViewportHeight > 0 ? Math.max(0, params.agentsViewportHeight - projectPx) : 0,
    availableHeight: Math.max(0, params.sectionsHeight - projectPx),
    collapsedAgentCount,
    extraChromePx: params.screenChromePx,
    groupRows: params.groupRows,
    machineCount: machineIds.size,
    openAgents,
    pinnedRows: params.pinnedRows
  })

  return { dynamicLimits, resolvedProjectsByAgent }
}

/** Sync backend `projects.db` records into `$portalAgentProjects` for every visible agent and
 *  re-sync the active owner whenever `$projects` updates (never copying unscoped `$projects` across profiles). */
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
    if (gatewayState !== 'open') {
      return
    }

    void syncPortalAgentProjects({ connectionId: activeConnection ?? null, profile: activeProfile || 'default' })
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
  const [sectionsEl, setSectionsEl] = useState<HTMLDivElement | null>(null)
  const [layoutMetrics, setLayoutMetrics] = useState({ agentsViewportHeight: 0, screenChromePx: 0, sectionsHeight: 0 })
  const chatOrder = useStore($portalAgentChatOrder)
  // Session-only stepped expansion (+10 per click) so one lookup never permanently blows up the sidebar.
  const [extraChatsByAgent, setExtraChatsByAgent] = useState<Record<string, number>>({})
  const allProjects = useStore($portalAgentProjects)
  const expandedProjects = useStore($portalProjectExpanded)
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

      const selectedOwner = knownSessionOwner(all, selectedStoredSessionId)
      const ownerMatches =
        !selectedOwner ||
        (typeof selectedOwner === 'string'
          ? selectedOwner === pendingProjectNewChat.owner.profile
          : agentSidebarKey({
              connectionId: selectedOwner.connectionId,
              profile: selectedOwner.targetProfile || selectedOwner.profile
            }) === agentSidebarKey(pendingProjectNewChat.owner))

      if (!ownerMatches) {
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
  }, [agents, all, pendingProjectNewChat, selectedStoredSessionId])
  const botModeLoaded = useBotModeLoaded()
  useEffect(() => {
    if (!sectionsEl || typeof ResizeObserver === 'undefined') {
      return
    }

    const findAgentsScroll = () =>
      sectionsEl.querySelector<HTMLElement>('[data-slot="agent-section"] .portal-section-scroll')

    const update = () => {
      const agentsScroll = findAgentsScroll()
      let screenPx = 0

      for (const screen of sectionsEl.querySelectorAll<HTMLElement>(
        '[data-slot="agent-screen-row"], .portal-agent-screen'
      )) {
        screenPx += Math.round(screen.getBoundingClientRect().height)
      }

      const next = {
        agentsViewportHeight: agentsScroll ? Math.round(agentsScroll.clientHeight) : 0,
        screenChromePx: screenPx,
        sectionsHeight: Math.round(sectionsEl.clientHeight)
      }
      setLayoutMetrics(prev =>
        Math.abs(prev.sectionsHeight - next.sectionsHeight) >= 8 ||
        Math.abs(prev.agentsViewportHeight - next.agentsViewportHeight) >= 8 ||
        Math.abs(prev.screenChromePx - next.screenChromePx) >= 8
          ? next
          : prev
      )
    }

    update()
    const observer = new ResizeObserver(update)
    observer.observe(sectionsEl)
    const agentsScroll = findAgentsScroll()

    if (agentsScroll) {
      observer.observe(agentsScroll)
    }

    return () => observer.disconnect()
  }, [agents.agents.length, botModeLoaded, pinned.length, sectionsEl])
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

    const { dynamicLimits, resolvedProjectsByAgent } = buildAgentChatLimits({
      activeConnection,
      activeProfile,
      agents,
      agentsViewportHeight: layoutMetrics.agentsViewportHeight,
      allProjects,
      expanded,
      expandedProjects,
      groupRows: groups ? groups.rooms.length + groups.names.length : 0,
      pinnedRows: pinned.length,
      screenChromePx: layoutMetrics.screenChromePx,
      sectionsHeight: layoutMetrics.sectionsHeight
    })
    return (
      <>
        {nothingFound && (
          <p className="portal-sidebar-empty" role="status">
            {t.sidebar.noMatch(query)}
          </p>
        )}
        <div className="portal-sidebar-sections" hidden={nothingFound} onKeyDown={moveRowFocus} ref={setSectionsEl}>
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
              const extraChats = extraChatsByAgent[key] ?? 0
              const baseLimit = dynamicLimits[key] ?? AGENT_CHAT_CAP
              const agentProjects = agentPortalProjectsFor(allProjects, owner)
              const ordered = orderAgentChats(sessions, chatOrder[key])
              const resolvedMap = resolvedProjectsByAgent.get(agentKey(agent))
              const sessionsByProject = new Map<string, typeof ordered>()
              const looseSessions: typeof ordered = []

              for (const row of ordered) {
                const projectId = resolvedMap ? (resolvedMap.get(row.id) ?? null) : resolvePortalProjectForSession(agentProjects, row)

                if (projectId) {
                  const list = sessionsByProject.get(projectId) ?? []
                  list.push(row)
                  sessionsByProject.set(projectId, list)
                } else {
                  looseSessions.push(row)
                }
              }

              const allAgentSessionIds = all
                .filter(
                  row =>
                    (row.connection_id ?? null) === (owner.connectionId ?? null) &&
                    (row.profile || 'default') === (owner.profile || 'default')
                )
                .map(row => row.id)

              return (
                <div className="portal-agent-sessions">
                  <PortalAgentProjects
                    agent={agent}
                    allAgentSessionIds={allAgentSessionIds}
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
                      extra: extraChats,
                      limit: baseLimit,
                      showAll: false,
                      onShowLess: () =>
                        setExtraChatsByAgent(current => {
                          const { [key]: _dropped, ...rest } = current

                          return rest
                        }),
                      onShowMore: () =>
                        setExtraChatsByAgent(current => {
                          const remaining = Math.max(0, looseSessions.length - (baseLimit + (current[key] ?? 0)))

                          return { ...current, [key]: (current[key] ?? 0) + computeShowMoreStep(remaining) }
                        }),
                      onToggle: () =>
                        setExtraChatsByAgent(current =>
                          (current[key] ?? 0) > 0
                            ? Object.fromEntries(Object.entries(current).filter(([k]) => k !== key))
                            : {
                                ...current,
                                [key]: Math.max(
                                  looseSessions.length - baseLimit,
                                  computeShowMoreStep(looseSessions.length - baseLimit)
                                )
                              }
                        )
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
      {botModeLoaded && <PortalBotAgentDialogs />}
      {/* One mount for the whole app. The header of WorktreeDialog tells why. */}
      <WorktreeDialog />
      <CreateAccountCard />
      <UsageStatsCard />
    </Sidebar>
  )
}
