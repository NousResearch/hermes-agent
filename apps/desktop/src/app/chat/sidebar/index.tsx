import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { writeClipboardText } from '@/components/ui/copy-button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { KbdGroup } from '@/components/ui/kbd'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem
} from '@/components/ui/sidebar'
import { Skeleton } from '@/components/ui/skeleton'
import {
  exportWorkflowProject,
  listWorkflowProjects,
  removeWorkflowProjectFromHistory,
  searchSessions,
  updateWorkflowProject,
  type SessionInfo,
  type SessionSearchResult
} from '@/hermes'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { workflowCopyFor } from '@/app/workflows/i18n'
import {
  $pinnedWorkflowProjectIds,
  $pinnedSessionIds,
  $sidebarAgentsGrouped,
  $sidebarOpen,
  $sidebarPinsOpen,
  $sidebarRecentsOpen,
  pinSession,
  pinWorkflowProject,
  reorderPinnedSession,
  setSidebarAgentsGrouped,
  setSidebarPinsOpen,
  setSidebarRecentsOpen,
  SIDEBAR_SESSIONS_PAGE_SIZE,
  unpinSession,
  unpinWorkflowProject
} from '@/store/layout'
import { notify, notifyError } from '@/store/notifications'
import {
  $selectedStoredSessionId,
  $sessions,
  $sessionsLoading,
  $sessionsTotal,
  $workingSessionIds,
  sessionPinId
} from '@/store/session'
import { $workflowLanguage } from '@/store/workflow-language'
import type { WorkflowProject } from '@/types/workflow'

import { type AppView, ARTIFACTS_ROUTE, MESSAGING_ROUTE, SKILLS_ROUTE, WORKFLOWS_ROUTE } from '../../routes'
import { SidebarPanelLabel } from '../../shell/sidebar-label'
import type { SidebarNavItem } from '../../types'

import { SidebarSessionRow } from './session-row'
import { VirtualSessionList } from './virtual-session-list'

const VIRTUALIZE_THRESHOLD = 25

// Render the modifier key the user actually presses on this platform. The
// global accelerator is bound to both Cmd+N (macOS) and Ctrl+N (everywhere
// else) in desktop-controller.tsx, but the hint should match muscle memory.
const NEW_SESSION_KBD: readonly string[] =
  typeof navigator !== 'undefined' && navigator.platform.toLowerCase().includes('mac') ? ['⌘', 'N'] : ['Ctrl', 'N']

const SIDEBAR_NAV: SidebarNavItem[] = [
  {
    id: 'new-session',
    label: 'New chat',
    icon: props => <Codicon name="robot" {...props} />,
    action: 'new-session'
  },
  {
    id: 'workflows',
    label: 'Workflow Workbench',
    icon: props => <Codicon name="graph" {...props} />,
    route: `${WORKFLOWS_ROUTE}?new=1`
  },
  {
    id: 'skills',
    label: 'Skills & Tools',
    icon: props => <Codicon name="symbol-misc" {...props} />,
    route: SKILLS_ROUTE
  },
  { id: 'messaging', label: 'Messaging', icon: props => <Codicon name="comment" {...props} />, route: MESSAGING_ROUTE },
  { id: 'artifacts', label: 'Artifacts', icon: props => <Codicon name="files" {...props} />, route: ARTIFACTS_ROUTE }
]

const WORKSPACE_PAGE = 5
const WS_ID_PREFIX = 'workspace:'
const WORKFLOW_SESSION_PREFIXES = ['workflow-project-', 'workflow-node-'] as const

const wsId = (id: string) => `${WS_ID_PREFIX}${id}`
const parseWsId = (id: string) => (id.startsWith(WS_ID_PREFIX) ? id.slice(WS_ID_PREFIX.length) : null)
const countLabel = (loaded: number, total: number) => (total > loaded ? `${loaded}/${total}` : String(loaded))
const sessionTime = (s: SessionInfo) => s.last_active || s.started_at || 0
const isWorkflowSessionId = (id: null | string | undefined) =>
  WORKFLOW_SESSION_PREFIXES.some(prefix => String(id ?? '').startsWith(prefix))
const isWorkflowSession = (session: Pick<SessionInfo, 'id' | 'source' | '_lineage_root_id'>) =>
  session.source === 'workflow' || isWorkflowSessionId(session.id) || isWorkflowSessionId(session._lineage_root_id)

const relativeProjectTime = (project: WorkflowProject) => {
  const timestamp = project.lastOpenedAt ?? project.updatedAt ?? project.createdAt
  const seconds = Math.max(0, Date.now() / 1000 - timestamp)

  if (seconds < 60) {
    return 'now'
  }

  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m`
  }

  if (seconds < 86400) {
    return `${Math.floor(seconds / 3600)}h`
  }

  return `${Math.floor(seconds / 86400)}d`
}

function orderByIds<T>(items: T[], getId: (item: T) => string, orderIds: string[]): T[] {
  if (!orderIds.length) {
    return items
  }

  const byId = new Map(items.map(item => [getId(item), item]))
  const seen = new Set<string>()
  const out: T[] = []

  for (const id of orderIds) {
    const item = byId.get(id)

    if (item) {
      out.push(item)
      seen.add(id)
    }
  }

  for (const item of items) {
    if (!seen.has(getId(item))) {
      out.push(item)
    }
  }

  return out
}

const baseName = (path: string) =>
  path
    .replace(/[/\\]+$/, '')
    .split(/[/\\]/)
    .filter(Boolean)
    .pop()

// FTS results cover sessions that aren't in the loaded page; synthesize a
// minimal SessionInfo so they render in the same row component (resume works
// by id; the snippet stands in for the preview).
function searchResultToSession(result: SessionSearchResult): SessionInfo {
  const ts = result.session_started ?? Date.now() / 1000

  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: result.session_id,
    input_tokens: 0,
    is_active: false,
    last_active: ts,
    message_count: 0,
    model: result.model ?? null,
    output_tokens: 0,
    preview: result.snippet?.trim() || null,
    source: result.source ?? null,
    started_at: ts,
    title: null,
    tool_call_count: 0
  }
}

function workspaceGroupsFor(sessions: SessionInfo[]): SidebarSessionGroup[] {
  const groups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    const path = session.cwd?.trim() || ''
    const id = path || '__no_workspace__'
    const label = baseName(path) || path || 'No workspace'

    const group = groups.get(id) ?? { id, label, path: path || null, sessions: [] }
    group.sessions.push(session)
    groups.set(id, group)
  }

  // Groups keep recency order (Map insertion = first-seen in the recency-sorted
  // input, so an active project floats up), but rows *within* a group sort by
  // creation time so they don't reshuffle every time a message lands — keeps
  // muscle memory intact.
  for (const group of groups.values()) {
    group.sessions.sort((a, b) => b.started_at - a.started_at)
  }

  return [...groups.values()]
}

function useSortableBindings(id: string) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id })

  return {
    dragging: isDragging,
    dragHandleProps: { ...attributes, ...listeners },
    ref: setNodeRef,
    reorderable: true as const,
    style: { transform: CSS.Transform.toString(transform), transition }
  }
}

interface ChatSidebarProps extends React.ComponentProps<typeof Sidebar> {
  currentView: AppView
  onNavigate: (item: SidebarNavItem) => void
  onLoadMoreSessions: () => void
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onNewSessionInWorkspace: (path: null | string) => void
}

export function ChatSidebar({
  currentView,
  onNavigate,
  onLoadMoreSessions,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onNewSessionInWorkspace
}: ChatSidebarProps) {
  const sidebarOpen = useStore($sidebarOpen)
  const location = useLocation()
  const agentsGrouped = useStore($sidebarAgentsGrouped)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const pinnedWorkflowProjectIds = useStore($pinnedWorkflowProjectIds)
  const workflowCopy = workflowCopyFor(useStore($workflowLanguage))
  const pinsOpen = useStore($sidebarPinsOpen)
  const agentsOpen = useStore($sidebarRecentsOpen)
  const selectedSessionId = useStore($selectedStoredSessionId)
  const sessions = useStore($sessions)
  const sessionsLoading = useStore($sessionsLoading)
  const sessionsTotal = useStore($sessionsTotal)
  const workingSessionIds = useStore($workingSessionIds)
  const [agentOrderIds, setAgentOrderIds] = useState<string[]>([])
  const [workspaceOrderIds, setWorkspaceOrderIds] = useState<string[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [workflowProjects, setWorkflowProjects] = useState<WorkflowProject[]>([])
  const [workflowProjectsLoading, setWorkflowProjectsLoading] = useState(false)
  const [workflowProjectsOpen, setWorkflowProjectsOpen] = useState(true)
  const trimmedQuery = searchQuery.trim()

  const activeSidebarSessionId = currentView === 'chat' ? selectedSessionId : null
  const currentWorkflowProjectId = useMemo(() => {
    if (currentView !== 'workflows') {
      return null
    }

    return new URLSearchParams(location.search).get('project')
  }, [currentView, location.search])

  const navigateToHermesHome = () =>
    onNavigate({
      action: 'new-session',
      icon: props => <Codicon name="robot" {...props} />,
      id: 'new-session',
      label: 'New chat'
    })

  const dndSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  const sortedSessions = useMemo(
    () => sessions.filter(session => !isWorkflowSession(session)).sort((a, b) => sessionTime(b) - sessionTime(a)),
    [sessions]
  )

  const sortedWorkflowProjects = useMemo(() => {
    const pinOrder = new Map(pinnedWorkflowProjectIds.map((id, index) => [id, index]))

    return workflowProjects
      .filter(project => !project.archived)
      .sort((a, b) => {
        const aPin = pinOrder.get(a.id)
        const bPin = pinOrder.get(b.id)
        if (aPin !== undefined || bPin !== undefined) {
          if (aPin === undefined) return 1
          if (bPin === undefined) return -1
          return aPin - bPin
        }

        return (b.lastOpenedAt ?? b.updatedAt) - (a.lastOpenedAt ?? a.updatedAt)
      })
  }, [pinnedWorkflowProjectIds, workflowProjects])

  const pinnedWorkflowProjectIdSet = useMemo(() => new Set(pinnedWorkflowProjectIds), [pinnedWorkflowProjectIds])

  const workingSessionIdSet = useMemo(() => new Set(workingSessionIds), [workingSessionIds])

  useEffect(() => {
    if (!sidebarOpen) {
      return
    }

    let disposed = false
    setWorkflowProjectsLoading(true)
    void listWorkflowProjects()
      .then(result => {
        if (!disposed) {
          setWorkflowProjects(result.projects)
        }
      })
      .catch(() => {
        if (!disposed) {
          setWorkflowProjects([])
        }
      })
      .finally(() => {
        if (!disposed) {
          setWorkflowProjectsLoading(false)
        }
      })

    return () => {
      disposed = true
    }
  }, [sidebarOpen])

  // Index sessions by both their live id and their lineage-root id so a pin
  // stored as the pre-compression root resolves to the live continuation tip.
  const sessionByAnyId = useMemo(() => {
    const map = new Map<string, SessionInfo>()

    for (const s of sortedSessions) {
      map.set(s.id, s)

      if (s._lineage_root_id && !map.has(s._lineage_root_id)) {
        map.set(s._lineage_root_id, s)
      }
    }

    return map
  }, [sortedSessions])

  const pinnedSessions = useMemo(() => {
    const seen = new Set<string>()
    const out: SessionInfo[] = []

    for (const pinId of pinnedSessionIds) {
      const session = sessionByAnyId.get(pinId)

      if (session && !seen.has(session.id)) {
        seen.add(session.id)
        out.push(session)
      }
    }

    return out
  }, [pinnedSessionIds, sessionByAnyId])

  const pinnedRealIdSet = useMemo(() => new Set(pinnedSessions.map(s => s.id)), [pinnedSessions])

  // Full-text search across *all* sessions (not just the loaded page) so 699
  // sessions stay findable. Debounced; loaded sessions are matched instantly
  // client-side and merged ahead of the server hits.
  useEffect(() => {
    if (!trimmedQuery) {
      setServerMatches([])

      return
    }

    let cancelled = false

    const id = window.setTimeout(() => {
      void searchSessions(trimmedQuery)
        .then(res => {
          if (!cancelled) {
            setServerMatches(res.results)
          }
        })
        .catch(() => undefined)
    }, 200)

    return () => {
      cancelled = true
      window.clearTimeout(id)
    }
  }, [trimmedQuery])

  const searchResults = useMemo(() => {
    if (!trimmedQuery) {
      return []
    }

    const needle = trimmedQuery.toLowerCase()
    const out = new Map<string, SessionInfo>()

    for (const s of sortedSessions) {
      if (`${s.title ?? ''} ${s.preview ?? ''} ${s.cwd ?? ''}`.toLowerCase().includes(needle)) {
        out.set(s.id, s)
      }
    }

    for (const match of serverMatches) {
      if (match.source === 'workflow' || isWorkflowSessionId(match.session_id)) {
        continue
      }

      if (out.has(match.session_id)) {
        continue
      }

      const loaded = sessionByAnyId.get(match.session_id)
      out.set(match.session_id, loaded ?? searchResultToSession(match))
    }

    return [...out.values()]
  }, [trimmedQuery, sortedSessions, serverMatches, sessionByAnyId])

  const unpinnedAgentSessions = useMemo(
    () => sortedSessions.filter(s => !pinnedRealIdSet.has(s.id)),
    [sortedSessions, pinnedRealIdSet]
  )

  const agentSessions = useMemo(
    () => orderByIds(unpinnedAgentSessions, s => s.id, agentOrderIds),
    [unpinnedAgentSessions, agentOrderIds]
  )

  const agentGroups = useMemo(
    () => orderByIds(workspaceGroupsFor(agentSessions), g => g.id, workspaceOrderIds),
    [agentSessions, workspaceOrderIds]
  )

  const showSessionSkeletons = sessionsLoading && sortedSessions.length === 0
  const showSessionSections = showSessionSkeletons || sortedSessions.length > 0
  const knownSessionTotal = Math.max(sessionsTotal, sortedSessions.length)
  const hasMoreSessions = knownSessionTotal > sortedSessions.length
  const remainingSessionCount = Math.max(0, knownSessionTotal - sortedSessions.length)

  const handlePinnedDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) {
      return
    }

    const newIndex = pinnedSessions.findIndex(s => s.id === String(over.id))

    if (newIndex < 0) {
      return
    }

    // Sortable ids are live session ids; the pinned store is keyed by durable
    // (lineage-root) ids, so translate before reordering.
    const dragged = sessionByAnyId.get(String(active.id))
    reorderPinnedSession(dragged ? sessionPinId(dragged) : String(active.id), newIndex)
  }

  const handleAgentDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) {
      return
    }

    const activeId = String(active.id)
    const overId = String(over.id)
    const activeWs = parseWsId(activeId)
    const overWs = parseWsId(overId)

    if (activeWs && overWs) {
      const oldIdx = agentGroups.findIndex(g => g.id === activeWs)
      const newIdx = agentGroups.findIndex(g => g.id === overWs)

      if (oldIdx < 0 || newIdx < 0) {
        return
      }

      setWorkspaceOrderIds(arrayMove(agentGroups, oldIdx, newIdx).map(g => g.id))

      return
    }

    if (activeWs || overWs) {
      return
    }

    const oldIdx = agentSessions.findIndex(s => s.id === activeId)
    const newIdx = agentSessions.findIndex(s => s.id === overId)

    if (oldIdx < 0 || newIdx < 0) {
      return
    }

    setAgentOrderIds(arrayMove(agentSessions, oldIdx, newIdx).map(s => s.id))
  }

  return (
    <Sidebar
      className={cn(
        'relative h-full min-w-0 overflow-hidden border-r border-t-0 border-b-0 border-l-0 text-foreground transition-none',
        sidebarOpen
          ? 'border-(--sidebar-edge-border) bg-(--ui-sidebar-surface-background) opacity-100'
          : 'pointer-events-none border-transparent bg-transparent opacity-0'
      )}
      collapsible="none"
    >
      <SidebarContent className="gap-0 overflow-hidden bg-transparent px-2.5">
        <SidebarGroup className="shrink-0 p-0 pb-2 pt-[calc(var(--titlebar-height)+0.375rem)]">
          <SidebarGroupContent>
            <SidebarMenu className="gap-px">
              {SIDEBAR_NAV.map(item => {
                const isInteractive = Boolean(item.action) || Boolean(item.route)

                const active =
                  (item.id === 'new-session' && currentView === 'chat') ||
                  (item.id === 'workflows' && currentView === 'workflows') ||
                  (item.id === 'skills' && currentView === 'skills') ||
                  (item.id === 'messaging' && currentView === 'messaging') ||
                  (item.id === 'artifacts' && currentView === 'artifacts')

                return (
                  <SidebarMenuItem key={item.id}>
                    <SidebarMenuButton
                      aria-disabled={!isInteractive}
                      className={cn(
                        'flex h-7 w-full cursor-pointer justify-start gap-2 rounded-md border border-transparent px-2 text-left text-[0.8125rem] font-medium text-(--ui-text-secondary) transition-colors duration-100 ease-out hover:bg-(--ui-control-hover-background) hover:text-foreground hover:transition-none',
                        active &&
                          'border-(--ui-stroke-tertiary) bg-(--ui-control-active-background) text-foreground shadow-none hover:border-(--ui-stroke-tertiary)!',
                        !isInteractive &&
                          'cursor-default hover:border-transparent hover:bg-transparent hover:text-inherit'
                      )}
                      onClick={() =>
                        onNavigate(item.id === 'workflows' ? { ...item, label: workflowCopy.workflowSidebarLabel } : item)
                      }
                      tooltip={item.id === 'workflows' ? workflowCopy.workflowSidebarLabel : item.label}
                      type="button"
                    >
                      <item.icon className="size-4 shrink-0 text-[color-mix(in_srgb,currentColor_72%,transparent)]" />
                      {sidebarOpen && (
                        <>
                          <span className="min-w-0 flex-1 truncate max-[46.25rem]:hidden">
                            {item.id === 'workflows' ? workflowCopy.workflowSidebarLabel : item.label}
                          </span>
                          {item.id === 'new-session' && (
                            <KbdGroup className="ml-auto max-[46.25rem]:hidden" keys={[...NEW_SESSION_KBD]} />
                          )}
                        </>
                      )}
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {sidebarOpen && showSessionSections && (
          <div className="shrink-0 pb-1 pt-1">
            <div className="flex items-center gap-1.5 rounded-md border border-transparent bg-transparent px-2 transition-colors focus-within:border-(--ui-stroke-tertiary)">
              <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="search" size="0.75rem" />
              <input
                aria-label="Search sessions"
                className="h-6 min-w-0 flex-1 bg-transparent text-[0.8125rem] text-foreground placeholder:text-(--ui-text-tertiary) focus:outline-none"
                onChange={event => setSearchQuery(event.target.value)}
                placeholder="Search sessions…"
                type="text"
                value={searchQuery}
              />
              {searchQuery && (
                <button
                  aria-label="Clear search"
                  className="grid size-4 shrink-0 cursor-pointer place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-active-background) hover:text-foreground"
                  onClick={() => setSearchQuery('')}
                  type="button"
                >
                  <Codicon name="close" size="0.75rem" />
                </button>
              )}
            </div>
          </div>
        )}

        {sidebarOpen && showSessionSections && trimmedQuery && (
          <SidebarSessionsSection
            activeSessionId={activeSidebarSessionId}
            contentClassName="flex min-h-0 flex-1 flex-col gap-px overflow-y-auto overscroll-contain pb-1.75"
            emptyState={
              <div className="grid min-h-24 place-items-center rounded-lg px-2 text-center text-xs text-(--ui-text-tertiary)">
                No sessions match “{trimmedQuery}”.
              </div>
            }
            label="Results"
            labelMeta={String(searchResults.length)}
            onArchiveSession={onArchiveSession}
            onDeleteSession={onDeleteSession}
            onResumeSession={onResumeSession}
            onToggle={() => undefined}
            onTogglePin={pinSession}
            open
            pinned={false}
            rootClassName="min-h-0 flex-1 p-0"
            sessions={searchResults}
            workingSessionIdSet={workingSessionIdSet}
          />
        )}

        {sidebarOpen && !trimmedQuery && (
          <SidebarWorkflowProjectsSection
            loading={workflowProjectsLoading}
            onOpenProject={projectId =>
              onNavigate({
                id: 'workflows',
                label: workflowCopy.workflowSidebarLabel,
                route: `${WORKFLOWS_ROUTE}?project=${encodeURIComponent(projectId)}`,
                icon: props => <Codicon name="graph" {...props} />
              })
            }
            onProjectRemoved={projectId => {
              setWorkflowProjects(projects => projects.filter(project => project.id !== projectId))
              unpinWorkflowProject(projectId)
              if (currentWorkflowProjectId === projectId) {
                navigateToHermesHome()
              }
            }}
            onProjectUpdated={project => {
              setWorkflowProjects(projects => {
                if (project.archived) {
                  return projects.filter(item => item.id !== project.id)
                }

                return projects.some(item => item.id === project.id)
                  ? projects.map(item => (item.id === project.id ? project : item))
                  : [project, ...projects]
              })

              if (project.archived && currentWorkflowProjectId === project.id) {
                navigateToHermesHome()
              }
            }}
            onTogglePin={projectId => {
              if (pinnedWorkflowProjectIdSet.has(projectId)) {
                unpinWorkflowProject(projectId)
              } else {
                pinWorkflowProject(projectId)
              }
            }}
            onToggle={() => setWorkflowProjectsOpen(value => !value)}
            open={workflowProjectsOpen}
            pinnedProjectIds={pinnedWorkflowProjectIdSet}
            projects={sortedWorkflowProjects}
            emptyLabel={workflowCopy.workflowProjectsEmpty}
          />
        )}

        {sidebarOpen && showSessionSections && !trimmedQuery && (
          <SidebarSessionsSection
            activeSessionId={activeSidebarSessionId}
            contentClassName="flex min-h-10 shrink-0 flex-col gap-px rounded-lg pb-2 pt-1"
            dndSensors={dndSensors}
            emptyState={<SidebarPinnedEmptyState />}
            label="Pinned"
            onArchiveSession={onArchiveSession}
            onDeleteSession={onDeleteSession}
            onReorder={handlePinnedDragEnd}
            onResumeSession={onResumeSession}
            onToggle={() => setSidebarPinsOpen(!pinsOpen)}
            onTogglePin={unpinSession}
            open={pinsOpen}
            pinned
            rootClassName="shrink-0 p-0 pb-1"
            sessions={pinnedSessions}
            sortable={pinnedSessions.length > 1}
            workingSessionIdSet={workingSessionIdSet}
          />
        )}

        {sidebarOpen && showSessionSections && !trimmedQuery && (
          <SidebarSessionsSection
            activeSessionId={activeSidebarSessionId}
            contentClassName="flex min-h-0 flex-1 flex-col gap-px overflow-y-auto overscroll-contain pb-1.75"
            dndSensors={dndSensors}
            emptyState={showSessionSkeletons ? <SidebarSessionSkeletons /> : <SidebarAllPinnedState />}
            footer={
              !agentsGrouped && !showSessionSkeletons && hasMoreSessions ? (
                <SidebarLoadMoreRow
                  loading={sessionsLoading}
                  onClick={onLoadMoreSessions}
                  step={Math.min(SIDEBAR_SESSIONS_PAGE_SIZE, remainingSessionCount)}
                />
              ) : null
            }
            forceEmptyState={showSessionSkeletons}
            groups={agentsGrouped ? agentGroups : undefined}
            headerAction={
              // Grouping operates on unpinned recents; if everything is
              // pinned the toggle does nothing visible, so hide it to avoid
              // a phantom click target.
              agentSessions.length > 0 ? (
                <Button
                  aria-label={agentsGrouped ? 'Show sessions as a single list' : 'Group sessions by workspace'}
                  className={cn(
                    'cursor-pointer text-(--ui-text-tertiary) opacity-70 hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100',
                    agentsGrouped && 'bg-(--ui-control-active-background) text-foreground opacity-100'
                  )}
                  onClick={event => {
                    event.stopPropagation()
                    setSidebarRecentsOpen(true)
                    setSidebarAgentsGrouped(!agentsGrouped)
                  }}
                  size="icon-xs"
                  title={agentsGrouped ? 'Ungroup sessions' : 'Group by workspace'}
                  variant="ghost"
                >
                  <Codicon name={agentsGrouped ? 'list-unordered' : 'root-folder'} size="0.75rem" />
                </Button>
              ) : null
            }
            label="Sessions"
            labelMeta={countLabel(agentSessions.length, knownSessionTotal)}
            onArchiveSession={onArchiveSession}
            onDeleteSession={onDeleteSession}
            onNewSessionInWorkspace={onNewSessionInWorkspace}
            onReorder={handleAgentDragEnd}
            onResumeSession={onResumeSession}
            onToggle={() => setSidebarRecentsOpen(!agentsOpen)}
            onTogglePin={pinSession}
            open={agentsOpen}
            pinned={false}
            rootClassName="min-h-0 flex-1 p-0"
            sessions={agentSessions}
            sortable={agentSessions.length > 1}
            workingSessionIdSet={workingSessionIdSet}
          />
        )}
      </SidebarContent>
    </Sidebar>
  )
}

interface SidebarSectionHeaderProps {
  label: string
  open: boolean
  onToggle: () => void
  action?: React.ReactNode
  meta?: React.ReactNode
}

function SidebarSectionHeader({ label, open, onToggle, action, meta }: SidebarSectionHeaderProps) {
  return (
    <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
      <button
        className="group/section-label flex w-fit cursor-pointer items-center gap-1 bg-transparent text-left leading-none"
        onClick={onToggle}
        type="button"
      >
        <SidebarPanelLabel>{label}</SidebarPanelLabel>
        {meta && <SidebarCount>{meta}</SidebarCount>}
        <DisclosureCaret
          className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
          open={open}
        />
      </button>
      {action}
    </div>
  )
}

function SidebarSessionSkeletons() {
  return (
    <div aria-hidden="true" className="grid gap-px">
      {['w-32', 'w-40', 'w-28', 'w-36', 'w-24'].map((width, i) => (
        <div className="grid min-h-7 grid-cols-[minmax(0,1fr)_1.5rem] items-center rounded-lg" key={`${width}-${i}`}>
          <Skeleton className={cn('h-3.5 rounded-full', width)} />
          <Skeleton className="mx-auto size-4 rounded-md opacity-60" />
        </div>
      ))}
    </div>
  )
}

const SidebarAllPinnedState = () => (
  <div className="grid min-h-24 place-items-center rounded-lg text-center text-xs text-(--ui-text-tertiary)">
    Everything here is pinned. Unpin a chat to show it in recents.
  </div>
)

function SidebarPinnedEmptyState() {
  return (
    <div className="flex min-h-7 items-center gap-1.5 rounded-lg pl-2 text-[0.75rem] text-(--ui-text-tertiary)">
      <span className="grid w-3.5 shrink-0 place-items-center text-(--ui-text-quaternary)">
        <Codicon name="pin" size="0.75rem" />
      </span>
      <span>Shift-click a chat to pin · drag to reorder</span>
    </div>
  )
}

function SidebarWorkflowProjectsSection({
  emptyLabel,
  loading,
  onOpenProject,
  onProjectRemoved,
  onProjectUpdated,
  onTogglePin,
  onToggle,
  open,
  pinnedProjectIds,
  projects
}: {
  emptyLabel: string
  loading: boolean
  onOpenProject: (projectId: string) => void
  onProjectRemoved: (projectId: string) => void
  onProjectUpdated: (project: WorkflowProject) => void
  onTogglePin: (projectId: string) => void
  onToggle: () => void
  open: boolean
  pinnedProjectIds: Set<string>
  projects: WorkflowProject[]
}) {
  return (
    <SidebarGroup className="shrink-0 p-0 pb-1">
      <SidebarSectionHeader label="Workflows" meta={loading ? '...' : String(projects.length)} onToggle={onToggle} open={open} />
      {open && (
        <SidebarGroupContent className="flex max-h-44 min-h-0 flex-col gap-px overflow-y-auto rounded-lg pb-2 pt-1">
          {loading && projects.length === 0 ? (
            <SidebarSessionSkeletons />
          ) : projects.length ? (
            projects.slice(0, 16).map(project => (
              <WorkflowProjectContextMenu
                key={project.id}
                onProjectRemoved={onProjectRemoved}
                onProjectUpdated={onProjectUpdated}
                onTogglePin={onTogglePin}
                pinned={pinnedProjectIds.has(project.id)}
                project={project}
              >
                <button
                  className="group flex min-h-8 w-full cursor-pointer items-center gap-2 rounded-md px-2 text-left text-[0.8125rem] text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
                  onClick={() => onOpenProject(project.id)}
                  title={project.root}
                  type="button"
                >
                  <Codicon className="size-4 shrink-0 text-[color-mix(in_srgb,currentColor_72%,transparent)]" name={pinnedProjectIds.has(project.id) ? 'pin' : 'graph'} />
                  <span className="min-w-0 flex-1 truncate">{project.name}</span>
                  <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">{relativeProjectTime(project)}</span>
                </button>
              </WorkflowProjectContextMenu>
            ))
          ) : (
            <div className="rounded-md px-2 py-2 text-xs text-(--ui-text-tertiary)">{emptyLabel}</div>
          )}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

function WorkflowProjectContextMenu({
  children,
  onProjectRemoved,
  onProjectUpdated,
  onTogglePin,
  pinned,
  project
}: {
  children: React.ReactNode
  onProjectRemoved: (projectId: string) => void
  onProjectUpdated: (project: WorkflowProject) => void
  onTogglePin: (projectId: string) => void
  pinned: boolean
  project: WorkflowProject
}) {
  const [renameOpen, setRenameOpen] = useState(false)

  const archiveProject = async () => {
    triggerHaptic('selection')
    try {
      const bundle = await updateWorkflowProject(project.id, { archived: true })
      onProjectUpdated(bundle.project)
      notify({ durationMs: 2_000, kind: 'success', message: 'Workflow archived' })
    } catch (err) {
      notifyError(err, 'Archive workflow failed')
    }
  }

  const removeProject = async () => {
    triggerHaptic('warning')
    const confirmed = window.confirm(`Remove "${project.name}" from Workflow history?\n\nThe project folder will not be deleted.`)
    if (!confirmed) {
      return
    }
    onProjectRemoved(project.id)
    try {
      await removeWorkflowProjectFromHistory(project.id)
      notify({ durationMs: 2_000, kind: 'success', message: 'Workflow removed from history' })
    } catch (err) {
      onProjectUpdated(project)
      if (pinned) {
        onTogglePin(project.id)
      }
      notifyError(err, 'Remove workflow failed')
    }
  }

  const exportProject = async () => {
    triggerHaptic('selection')
    try {
      const result = await exportWorkflowProject(project)
      if (!result.canceled) {
        notify({ durationMs: 2_500, kind: 'success', message: 'Workflow exported' })
      }
    } catch (err) {
      notifyError(err, 'Export workflow failed')
    }
  }

  const copyText = async (value: string, label: string) => {
    triggerHaptic('selection')
    try {
      await writeClipboardText(value)
      notify({ durationMs: 1_500, kind: 'success', message: `${label} copied` })
    } catch (err) {
      notifyError(err, `Could not copy ${label.toLowerCase()}`)
    }
  }

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
        <ContextMenuContent aria-label={`Actions for ${project.name}`} className="w-48">
          <ContextMenuItem onSelect={() => onTogglePin(project.id)}>
            <Codicon name="pin" size="0.875rem" />
            <span>{pinned ? 'Unpin' : 'Pin'}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={event => {
            event.preventDefault()
            void copyText(project.id, 'Workflow ID')
          }}>
            <Codicon name="copy" size="0.875rem" />
            <span>Copy ID</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={event => {
            event.preventDefault()
            void copyText(project.root, 'Workflow path')
          }}>
            <Codicon name="root-folder" size="0.875rem" />
            <span>Copy Path</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={() => void exportProject()}>
            <Codicon name="cloud-download" size="0.875rem" />
            <span>Export</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={() => setRenameOpen(true)}>
            <Codicon name="edit" size="0.875rem" />
            <span>Rename</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={() => void archiveProject()}>
            <Codicon name="archive" size="0.875rem" />
            <span>Archive</span>
          </ContextMenuItem>
          <ContextMenuItem className="text-destructive focus:text-destructive" onSelect={() => void removeProject()} variant="destructive">
            <Codicon name="trash" size="0.875rem" />
            <span>Remove from history</span>
          </ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>
      <RenameWorkflowProjectDialog
        currentName={project.name}
        onOpenChange={setRenameOpen}
        onProjectUpdated={onProjectUpdated}
        open={renameOpen}
        projectId={project.id}
      />
    </>
  )
}

function RenameWorkflowProjectDialog({
  currentName,
  onOpenChange,
  onProjectUpdated,
  open,
  projectId
}: {
  currentName: string
  onOpenChange: (open: boolean) => void
  onProjectUpdated: (project: WorkflowProject) => void
  open: boolean
  projectId: string
}) {
  const [value, setValue] = useState(currentName)
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(currentName)
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [currentName, open])

  const submit = async () => {
    const next = value.trim()
    if (!next || submitting) {
      return
    }
    if (next === currentName.trim()) {
      onOpenChange(false)
      return
    }
    setSubmitting(true)
    try {
      const bundle = await updateWorkflowProject(projectId, { name: next })
      onProjectUpdated(bundle.project)
      notify({ durationMs: 2_000, kind: 'success', message: 'Workflow renamed' })
      onOpenChange(false)
    } catch (err) {
      notifyError(err, 'Rename workflow failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Rename workflow</DialogTitle>
          <DialogDescription>Update the display name in the Workflow list. The folder path is not changed.</DialogDescription>
        </DialogHeader>
        <Input
          autoFocus
          disabled={submitting}
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              event.preventDefault()
              void submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            Cancel
          </Button>
          <Button disabled={submitting || !value.trim()} onClick={() => void submit()} type="button">
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface SidebarSessionGroup {
  id: string
  label: string
  path: null | string
  sessions: SessionInfo[]
}

interface SidebarSessionsSectionProps {
  label: string
  open: boolean
  onToggle: () => void
  sessions: SessionInfo[]
  activeSessionId: null | string
  workingSessionIdSet: Set<string>
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onTogglePin: (sessionId: string) => void
  onNewSessionInWorkspace?: (path: null | string) => void
  pinned: boolean
  rootClassName?: string
  contentClassName?: string
  emptyState: React.ReactNode
  forceEmptyState?: boolean
  headerAction?: React.ReactNode
  footer?: React.ReactNode
  groups?: SidebarSessionGroup[]
  labelMeta?: React.ReactNode
  sortable?: boolean
  onReorder?: (event: DragEndEvent) => void
  dndSensors?: ReturnType<typeof useSensors>
}

function SidebarSessionsSection({
  label,
  open,
  onToggle,
  sessions,
  activeSessionId,
  workingSessionIdSet,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onTogglePin,
  onNewSessionInWorkspace,
  pinned,
  rootClassName,
  contentClassName,
  emptyState,
  forceEmptyState = false,
  headerAction,
  footer,
  groups,
  labelMeta,
  sortable = false,
  onReorder,
  dndSensors
}: SidebarSessionsSectionProps) {
  const showEmptyState = forceEmptyState || sessions.length === 0
  const dndActive = sortable && !!onReorder

  const renderRow = (session: SessionInfo) => {
    const rowProps = {
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onDelete: () => onDeleteSession(session.id),
      onPin: () => onTogglePin(sessionPinId(session)),
      onResume: () => onResumeSession(session.id),
      session
    }

    return sortable ? (
      <SortableSidebarSessionRow key={session.id} {...rowProps} />
    ) : (
      <SidebarSessionRow key={session.id} {...rowProps} />
    )
  }

  const renderRows = (items: SessionInfo[]) => items.map(renderRow)

  const renderSessionList = (items: SessionInfo[]) =>
    dndActive ? (
      <SortableContext items={items.map(s => s.id)} strategy={verticalListSortingStrategy}>
        {renderRows(items)}
      </SortableContext>
    ) : (
      renderRows(items)
    )

  const flatVirtualized = !showEmptyState && !groups?.length && sessions.length >= VIRTUALIZE_THRESHOLD

  let inner: React.ReactNode

  if (showEmptyState) {
    inner = emptyState
  } else if (groups?.length) {
    const groupNodes = groups.map(group =>
      dndActive ? (
        <SortableSidebarWorkspaceGroup
          group={group}
          key={group.id}
          onNewSession={onNewSessionInWorkspace}
          renderRows={renderSessionList}
        />
      ) : (
        <SidebarWorkspaceGroup
          group={group}
          key={group.id}
          onNewSession={onNewSessionInWorkspace}
          renderRows={renderSessionList}
        />
      )
    )

    inner = dndActive ? (
      <SortableContext items={groups.map(g => wsId(g.id))} strategy={verticalListSortingStrategy}>
        {groupNodes}
      </SortableContext>
    ) : (
      groupNodes
    )
  } else if (flatVirtualized) {
    inner = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        onArchiveSession={onArchiveSession}
        onDeleteSession={onDeleteSession}
        onResumeSession={onResumeSession}
        onTogglePin={onTogglePin}
        pinned={pinned}
        sessions={sessions}
        sortable={sortable}
        workingSessionIdSet={workingSessionIdSet}
      />
    )
  } else {
    inner = renderSessionList(sessions)
  }

  const body =
    dndActive && !showEmptyState ? (
      <DndContext collisionDetection={closestCenter} onDragEnd={onReorder} sensors={dndSensors}>
        {inner}
      </DndContext>
    ) : (
      inner
    )

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

  return (
    <SidebarGroup className={rootClassName}>
      <SidebarSectionHeader action={headerAction} label={label} meta={labelMeta} onToggle={onToggle} open={open} />
      {open && (
        <SidebarGroupContent className={resolvedContentClassName}>
          {body}
          {footer}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

interface SidebarWorkspaceGroupProps extends React.ComponentProps<'div'> {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
}

function SidebarWorkspaceGroup({
  group,
  renderRows,
  onNewSession,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  className,
  style,
  ref,
  ...rest
}: SidebarWorkspaceGroupProps) {
  const [open, setOpen] = useState(true)
  const [visibleCount, setVisibleCount] = useState(WORKSPACE_PAGE)
  const visibleSessions = group.sessions.slice(0, visibleCount)
  const hiddenCount = Math.max(0, group.sessions.length - visibleSessions.length)
  const nextCount = Math.min(WORKSPACE_PAGE, hiddenCount)

  return (
    <div className={cn('grid gap-px', dragging && 'z-10 opacity-60', className)} ref={ref} style={style} {...rest}>
      <div className="group/workspace flex min-h-6 items-center gap-1 px-2 pt-1 text-[0.6875rem] font-medium text-(--ui-text-tertiary)">
        <button
          className="flex min-w-0 cursor-pointer items-center gap-1 bg-transparent text-left hover:text-(--ui-text-secondary)"
          onClick={() => setOpen(value => !value)}
          title={group.path ?? undefined}
          type="button"
        >
          <span className="truncate">{group.label}</span>
          <SidebarCount>{group.sessions.length}</SidebarCount>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
            open={open}
          />
        </button>
        {onNewSession && (
          <button
            aria-label={`New session in ${group.label}`}
            className="grid size-4 shrink-0 cursor-pointer place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100"
            onClick={() => onNewSession(group.path)}
            title={`New session in ${group.label}`}
            type="button"
          >
            <Codicon name="add" size="0.75rem" />
          </button>
        )}
        {reorderable && (
          <span
            {...dragHandleProps}
            aria-label={`Reorder workspace ${group.label}`}
            className="ml-auto -my-0.5 grid w-4 shrink-0 cursor-grab touch-none place-items-center self-stretch overflow-hidden active:cursor-grabbing"
            onClick={event => event.stopPropagation()}
          >
            <Codicon
              className={cn(
                'text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/workspace:opacity-80 hover:text-(--ui-text-secondary)',
                dragging && 'text-(--ui-text-secondary) opacity-100'
              )}
              name="grabber"
              size="0.75rem"
            />
          </span>
        )}
      </div>
      {open && (
        <>
          {renderRows(visibleSessions)}
          {hiddenCount > 0 && (
            <button
              aria-label={`Show ${nextCount} more in ${group.label}`}
              className="ml-auto grid size-5 cursor-pointer place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground"
              onClick={() => setVisibleCount(count => count + WORKSPACE_PAGE)}
              title={`Show ${nextCount} more in ${group.label}`}
              type="button"
            >
              <Codicon name="ellipsis" size="0.75rem" />
            </button>
          )}
        </>
      )}
    </div>
  )
}

interface SortableWorkspaceProps {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
}

function SortableSidebarWorkspaceGroup(props: SortableWorkspaceProps) {
  return <SidebarWorkspaceGroup {...props} {...useSortableBindings(wsId(props.group.id))} />
}

function SidebarCount({ children }: { children: React.ReactNode }) {
  return <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{children}</span>
}

interface SortableSessionRowProps {
  session: SessionInfo
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
}

function SortableSidebarSessionRow(props: SortableSessionRowProps) {
  return <SidebarSessionRow {...props} {...useSortableBindings(props.session.id)} />
}

interface SidebarLoadMoreRowProps {
  loading: boolean
  onClick: () => void
  step: number
}

function SidebarLoadMoreRow({ loading, onClick, step }: SidebarLoadMoreRowProps) {
  const label = loading ? 'Loading…' : step > 0 ? `Load ${step} more` : 'Load more'

  return (
    <button
      className="flex min-h-5 cursor-pointer items-center gap-1 self-start bg-transparent pl-2 text-left text-[0.6875rem] text-(--ui-text-tertiary) transition-colors duration-100 ease-out hover:text-foreground hover:transition-none disabled:cursor-default disabled:opacity-60 disabled:hover:text-(--ui-text-tertiary)"
      disabled={loading}
      onClick={onClick}
      type="button"
    >
      <Codicon className="opacity-70" name={loading ? 'loading' : 'chevron-down'} size="0.75rem" spinning={loading} />
      <span>{label}</span>
    </button>
  )
}
