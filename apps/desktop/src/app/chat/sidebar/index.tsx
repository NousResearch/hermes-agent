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
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
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
import type { SessionInfo } from '@/hermes'
import { cn } from '@/lib/utils'
import {
  $pinnedSessionIds,
  $sidebarAgentsGrouped,
  $sidebarOpen,
  $sidebarPinsOpen,
  $sidebarRecentsOpen,
  pinSession,
  reorderPinnedSession,
  setSidebarAgentsGrouped,
  setSidebarPinsOpen,
  setSidebarRecentsOpen,
  SIDEBAR_SESSIONS_PAGE_SIZE,
  unpinSession
} from '@/store/layout'
import {
  $selectedStoredSessionId,
  $sessions,
  $sessionsLoading,
  $sessionsTotal,
  $workingSessionIds
} from '@/store/session'

import { type AppView, ARTIFACTS_ROUTE, MESSAGING_ROUTE, SKILLS_ROUTE } from '../../routes'
import { SidebarPanelLabel } from '../../shell/sidebar-label'
import type { SidebarNavItem } from '../../types'

import { SidebarSessionRow } from './session-row'
import { VirtualSessionList } from './virtual-session-list'

const VIRTUALIZE_THRESHOLD = 25

const SIDEBAR_NAV: SidebarNavItem[] = [
  {
    id: 'new-session',
    label: 'New session',
    icon: props => <Codicon name="robot" {...props} />,
    action: 'new-session'
  },
  { id: 'skills', label: 'Skills', icon: props => <Codicon name="symbol-misc" {...props} />, route: SKILLS_ROUTE },
  { id: 'messaging', label: 'Messaging', icon: props => <Codicon name="comment" {...props} />, route: MESSAGING_ROUTE },
  { id: 'artifacts', label: 'Artifacts', icon: props => <Codicon name="files" {...props} />, route: ARTIFACTS_ROUTE }
]

const WORKSPACE_PAGE = 5
const WS_ID_PREFIX = 'workspace:'

const wsId = (id: string) => `${WS_ID_PREFIX}${id}`
const parseWsId = (id: string) => (id.startsWith(WS_ID_PREFIX) ? id.slice(WS_ID_PREFIX.length) : null)
const countLabel = (loaded: number, total: number) => (total > loaded ? `${loaded}/${total}` : String(loaded))
const sessionTime = (s: SessionInfo) => s.last_active || s.started_at || 0

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
  const agentsGrouped = useStore($sidebarAgentsGrouped)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const pinsOpen = useStore($sidebarPinsOpen)
  const agentsOpen = useStore($sidebarRecentsOpen)
  const selectedSessionId = useStore($selectedStoredSessionId)
  const sessions = useStore($sessions)
  const sessionsLoading = useStore($sessionsLoading)
  const sessionsTotal = useStore($sessionsTotal)
  const workingSessionIds = useStore($workingSessionIds)
  const [agentOrderIds, setAgentOrderIds] = useState<string[]>([])
  const [workspaceOrderIds, setWorkspaceOrderIds] = useState<string[]>([])

  const activeSidebarSessionId = currentView === 'chat' ? selectedSessionId : null

  const dndSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  const sortedSessions = useMemo(() => [...sessions].sort((a, b) => sessionTime(b) - sessionTime(a)), [sessions])

  const sessionsById = useMemo(() => new Map(sessions.map(s => [s.id, s])), [sessions])
  const workingSessionIdSet = useMemo(() => new Set(workingSessionIds), [workingSessionIds])

  const visiblePinnedIds = useMemo(
    () => pinnedSessionIds.filter(id => sessionsById.has(id)),
    [pinnedSessionIds, sessionsById]
  )

  const visiblePinnedIdSet = useMemo(() => new Set(visiblePinnedIds), [visiblePinnedIds])

  const pinnedSessions = useMemo(
    () => visiblePinnedIds.map(id => sessionsById.get(id)!).filter(Boolean),
    [visiblePinnedIds, sessionsById]
  )

  const unpinnedAgentSessions = useMemo(
    () => sortedSessions.filter(s => !visiblePinnedIdSet.has(s.id)),
    [sortedSessions, visiblePinnedIdSet]
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

    reorderPinnedSession(String(active.id), newIndex)
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
                      onClick={() => onNavigate(item)}
                      tooltip={item.label}
                      type="button"
                    >
                      <item.icon className="size-4 shrink-0 text-[color-mix(in_srgb,currentColor_72%,transparent)]" />
                      {sidebarOpen && (
                        <>
                          <span className="min-w-0 flex-1 truncate max-[46.25rem]:hidden">{item.label}</span>
                          {item.id === 'new-session' && (
                            <KbdGroup className="ml-auto max-[46.25rem]:hidden" keys={['⇧', 'N']} />
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

        {sidebarOpen && showSessionSections && (
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
      <span>Shift click to pin a chat</span>
    </div>
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
      onPin: () => onTogglePin(session.id),
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
