import type { useSensors } from '@dnd-kit/core'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo } from 'react'

import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { flattenSessionsWithBranches } from '@/lib/session-branch-tree'
import {
  groupEntriesByRecency,
  groupEntriesByStatus,
  hideCollapsedGroupRows,
  type SidebarListRow,
  toSessionRows
} from '@/lib/session-date-groups'
import { sessionBucketLabel } from '@/lib/time'
import { cn } from '@/lib/utils'
import {
  $sidebarListGroupIds,
  $sidebarWorkspaceNodeOpen,
  listGroupNodeId,
  toggleWorkspaceNodeCollapsed
} from '@/store/layout'
import { sessionPinId } from '@/store/session'
import { $sessionDotStateById, hasLiveTurn } from '@/store/session-dot-state'

import { SidebarDateDivider, SidebarSectionMeta } from './chrome'
import { mergeVisibleReorder, orderRowsWithinGroups, reorderableRowIds } from './order'
import {
  EnteredProjectContent,
  ProjectOverviewRow,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  SidebarWorkspaceGroup,
  type SidebarWorkspaceTree
} from './projects'
import { WorkspaceAddButton } from './projects/workspace-header'
import { ReorderableList, useSortableBindings } from './reorderable-list'
import { SidebarSessionSkeletons } from './section-states'
import { SidebarSessionRow } from './session-row'
import { VirtualSessionList } from './virtual-session-list'

export const VIRTUALIZE_THRESHOLD = 25

interface SidebarSectionHeaderProps {
  label: string
  open: boolean
  onToggle: () => void
  action?: React.ReactNode
  meta?: React.ReactNode
  icon?: React.ReactNode
  // When false the section can't be collapsed: the label renders static (no
  // toggle, no caret) and the section is always open. Used for the single-
  // project view, where collapsing one project makes no sense.
  collapsible?: boolean
}

function SidebarSectionHeader({
  label,
  open,
  onToggle,
  action,
  meta,
  icon,
  collapsible = true
}: SidebarSectionHeaderProps) {
  const labelBody = (
    <>
      {icon}
      <SidebarPanelLabel>{label}</SidebarPanelLabel>
      {meta && <SidebarSectionMeta>{meta}</SidebarSectionMeta>}
    </>
  )

  return (
    <div className="group/section flex shrink-0 items-center justify-between gap-1 pb-1 pt-1.5">
      {collapsible ? (
        <button
          // min-w-0 lets the label truncate at narrow sidebar widths instead of
          // pushing the header's trailing action icons out of view.
          className="group/section-label flex w-fit min-w-0 items-center gap-1 bg-transparent text-left leading-none"
          onClick={onToggle}
          type="button"
        >
          {labelBody}
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
      ) : (
        <div className="flex w-fit min-w-0 items-center gap-1 leading-none">{labelBody}</div>
      )}
      {action}
    </div>
  )
}

interface SidebarSessionsSectionProps {
  label: string
  open: boolean
  onToggle: () => void
  sessions: SessionInfo[]
  activeSessionId: null | string
  onResumeSession: (sessionId: string, session?: SessionInfo) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onBranchSession?: (sessionId: string, profile?: string) => void
  onTogglePin: (sessionId: string) => void
  onToggleUnread: (sessionId: string) => void
  onNewSessionInWorkspace?: (path: null | string) => void
  pinned: boolean
  rootClassName?: string
  contentClassName?: string
  emptyState: React.ReactNode
  forceEmptyState?: boolean
  headerAction?: React.ReactNode
  footer?: React.ReactNode
  groups?: SidebarSessionGroup[]
  tree?: SidebarWorkspaceTree[]
  // Project overview: when present, render a drill-in list of project rows
  // instead of sessions. Clicking a row enters that project (onEnterProject),
  // which then passes `projectContent` on the next render. Takes precedence
  // over `tree` / `groups`.
  projectOverview?: SidebarProjectTree[]
  // Per-project preview rows (from the backend tree), keyed by project id.
  projectOverviewPreviews?: Record<string, SessionInfo[]>
  // True while the backend project tree is loading (overview skeleton).
  projectsLoading?: boolean
  onEnterProject?: (id: string) => void
  // The entered project's flattened content: main-checkout sessions render
  // directly (no redundant repo/branch header); only linked worktrees nest.
  projectContent?: SidebarProjectTree
  // Live git lanes (`git worktree list`) for repos in the entered project —
  // a VISUAL enhancer only (empty lanes), never session membership.
  projectRepoWorktrees?: Record<string, HermesGitWorktree[]>
  // Live session cache used for optimistic placement inside entered-project lanes.
  liveSessions?: SessionInfo[]
  // Client-side optimistic eviction layer (deleted/archived ids).
  removedSessionIds?: ReadonlySet<string>
  activeProjectId?: null | string
  labelMeta?: React.ReactNode
  labelIcon?: React.ReactNode
  // When false the section header is static (no caret/toggle) and always open.
  collapsible?: boolean
  sortable?: boolean
  // The persisted drag order, applied WITHIN each date group (see
  // orderRowsWithinGroups). Chronology decides the groups; this decides the
  // sequence inside one, so a reorder no longer costs the whole list its
  // dividers. Pinned passes nothing — its rows arrive in pin order already.
  manualOrderIds?: string[]
  // The flat session list is the only hand-reorderable surface (grouped/project
  // views sort deterministically), so it owns the one ReorderableList.
  onReorderSessions?: (ids: string[]) => void
  // Drag-to-reorder for the project overview list (top-level projects).
  onReorderProjects?: (ids: string[]) => void
  // Rendered atop the entered-project body (a "back to overview" row).
  projectBackRow?: React.ReactNode
  dndSensors?: ReturnType<typeof useSensors>
  // Tag every row with its owning profile. Set on the flat cross-profile
  // lists (Pinned / search results) in the All-profiles view, where no group
  // header communicates ownership (#66003).
  showProfileTags?: boolean
  // Which dividers to fold into the flat list: `date` gives the chronological
  // "Yesterday" / "Last week" separators (flat recents + entered-project lanes),
  // `status` splits into WORKING / DONE under the same separators. `none` for
  // pinned, messaging groups, and the project overview, where the order isn't
  // strictly by recency so a bucket would be misleading.
  grouping?: 'date' | 'none' | 'status'
  // Inbox style: render every flat session row as a three-line card (project ·
  // age / title / model · size). A render variant that composes with whichever
  // grouping is active — the flat recents list opts in; dense tree surfaces
  // (pinned, projects, messaging) keep the one-line row.
  card?: boolean
}

export function SidebarSessionsSection({
  label,
  open,
  onToggle,
  sessions,
  activeSessionId,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onBranchSession,
  onTogglePin,
  onToggleUnread,
  onNewSessionInWorkspace,
  pinned,
  rootClassName,
  contentClassName,
  emptyState,
  forceEmptyState = false,
  headerAction,
  footer,
  groups,
  projectOverview,
  projectOverviewPreviews,
  projectsLoading = false,
  onEnterProject,
  projectContent,
  projectRepoWorktrees,
  liveSessions,
  removedSessionIds,
  activeProjectId,
  labelMeta,
  labelIcon,
  collapsible = true,
  sortable = false,
  manualOrderIds,
  onReorderSessions,
  onReorderProjects,
  projectBackRow,
  dndSensors,
  showProfileTags = false,
  grouping = 'none',
  card = false
}: SidebarSessionsSectionProps) {
  const { t } = useI18n()
  const dividerLabels = t.sidebar.dateDivider
  const statusDividerLabels = t.sidebar.statusDivider
  const dotStates = useStore($sessionDotStateById)
  const nodeOpen = useStore($sidebarWorkspaceNodeOpen)
  const isListGroupOpen = useCallback((key: string) => nodeOpen[listGroupNodeId(key)] ?? true, [nodeOpen])
  const sectionOpen = collapsible ? open : true
  const hasGroupedSessions = Boolean(groups?.some(group => group.sessions.length > 0))
  // A defined project list is itself content (even an empty project should
  // render as a drill-in row so the user can see it exists).
  const hasProjectOverview = Boolean(projectOverview?.length)

  // Lanes count as content even with no rows left in them: the backend only
  // emits a lane that has sessions, so a lane surviving with zero rows means
  // they were filtered out (pinned) — the branch is real and must still render.
  // A genuinely empty project has no lanes at all and keeps its empty state.
  const hasProjectContent = Boolean(
    projectContent && (projectContent.sessionCount > 0 || projectContent.repos.some(repo => repo.groups.length > 0))
  )

  const showEmptyState =
    forceEmptyState || (!hasGroupedSessions && !hasProjectOverview && !hasProjectContent && sessions.length === 0)

  // The flat recents/pinned list is the only place sessions reorder by hand;
  // grouped/tree views always sort by creation date and never drag.
  const sessionsDraggable = sortable && !!onReorderSessions

  // Only Pinned arrives pre-ordered as a flat sequence. Recents keeps its
  // recency sort — the drag order is layered on per date group below, so the
  // buckets stay truthful and a reorder never costs the list its dividers.
  const displayEntries = useMemo(
    () => flattenSessionsWithBranches(sessions, { preserveOrder: pinned }),
    [sessions, pinned]
  )

  const renderRow = useCallback(
    (session: SessionInfo, draggable: boolean, branchStem?: string) => {
      const rowProps = {
        branchStem,
        card,
        isPinned: pinned,
        isSelected: session.id === activeSessionId,
        onArchive: () => onArchiveSession(session.id),
        onBranch: onBranchSession ? () => onBranchSession(session.id, session.profile) : undefined,
        onDelete: () => onDeleteSession(session.id),
        onPin: () => onTogglePin(sessionPinId(session)),
        onToggleUnread: () => onToggleUnread(session.id),
        onResume: () => onResumeSession(session.id, session),
        reorderable: draggable && !branchStem,
        session,
        showProfile: showProfileTags,
        unread: session.unread === true
      }

      // Key by (profile, id): twins with the same stored id in two profiles
      // are distinct rows (#92454) — a bare-id key makes React misattribute
      // one twin's rendered state to the other.
      return draggable && !branchStem ? (
        <SortableSidebarSessionRow key={`${session.profile ?? ''}::${session.id}`} {...rowProps} />
      ) : (
        <SidebarSessionRow key={`${session.profile ?? ''}::${session.id}`} {...rowProps} />
      )
    },
    [
      activeSessionId,
      card,
      onArchiveSession,
      onBranchSession,
      onDeleteSession,
      onResumeSession,
      onTogglePin,
      onToggleUnread,
      pinned,
      showProfileTags
    ]
  )

  // Date dividers head a group the same way a repo header does, so they carry
  // the same hover-revealed "+". Only for dates: "new session in WORKING" is
  // not a thing.
  const dividerAction =
    grouping === 'date' && onNewSessionInWorkspace ? (
      <WorkspaceAddButton label={t.sidebar.nav['new-session']} onClick={() => onNewSessionInWorkspace(null)} />
    ) : null

  const dividerToggle = useMemo(
    () => ({
      ariaLabel: (label: string, open: boolean) => t.sidebar.projects.toggle(label, !open),
      onToggle: (key: string) => toggleWorkspaceNodeCollapsed(listGroupNodeId(key)),
      open: isListGroupOpen
    }),
    [isListGroupOpen, t]
  )

  // A single flat/virtual/lane list row — either a divider or a session.
  const renderListRow = useCallback(
    (row: SidebarListRow, draggable: boolean, action?: React.ReactNode) => {
      if (row.kind === 'session') {
        return renderRow(row.entry.session, draggable, row.entry.branchStem)
      }

      const label = 'label' in row ? row.label : sessionBucketLabel(row.bucket, dividerLabels)
      const open = dividerToggle.open(row.key)

      return (
        <SidebarDateDivider
          action={action}
          key={row.key}
          label={label}
          toggle={{
            ariaLabel: dividerToggle.ariaLabel(label, open),
            onToggle: () => dividerToggle.onToggle(row.key),
            open
          }}
        />
      )
    },
    [dividerLabels, dividerToggle, renderRow]
  )

  // Sessions inside repos/worktrees are date-ordered and static.
  const renderRows = useCallback(
    (items: SessionInfo[]) =>
      flattenSessionsWithBranches(items).map(({ branchStem, session }) => renderRow(session, false, branchStem)),
    [renderRow]
  )

  // Same as `renderRows`, but with date dividers folded in — used for
  // entered-project lanes so a lane spanning multiple days reads
  // chronologically, matching the flat recents list.
  const renderRowsDated = useCallback(
    (items: SessionInfo[]) => {
      const entries = flattenSessionsWithBranches(items)

      const rows = grouping === 'date' ? groupEntriesByRecency(entries) : toSessionRows(entries)

      return hideCollapsedGroupRows(rows, isListGroupOpen).map(row => renderListRow(row, false))
    },
    [grouping, isListGroupOpen, renderListRow]
  )

  // Flat recents as list rows: grouped by recency when enabled, plain otherwise.
  // The hand-picked order is then applied INSIDE each date group, so dragging a
  // row ranks it among its own day's chats instead of freezing the whole list
  // into an undated manual mode.
  const flatRows: SidebarListRow[] = useMemo(() => {
    const rows =
      grouping === 'date'
        ? groupEntriesByRecency(displayEntries)
        : grouping === 'status'
          ? groupEntriesByStatus(
              displayEntries,
              entry => hasLiveTurn(dotStates[entry.session.id] ?? 'idle'),
              statusDividerLabels
            )
          : toSessionRows(displayEntries)

    return manualOrderIds?.length ? orderRowsWithinGroups(rows, manualOrderIds) : rows
  }, [grouping, displayEntries, dotStates, manualOrderIds, statusDividerLabels])

  // Closed date/status buckets keep their divider and drop the sessions under
  // it. Same array when nothing is collapsed so the virtualizer's rows ref
  // stays stable across parent re-renders.
  const visibleRows = useMemo(
    () => hideCollapsedGroupRows(flatRows, isListGroupOpen),
    [flatRows, isListGroupOpen]
  )

  // dnd-kit must see exactly the ids it renders, in render order: the sortable
  // set is derived from the rows, not from `sessions`. Feeding it the unrendered
  // session order made a drop compute its target index against a list the user
  // wasn't looking at — the drag that landed a row in the wrong slot.
  const sortableRowIds = useMemo(() => reorderableRowIds(visibleRows), [visibleRows])
  const allSortableRowIds = useMemo(() => reorderableRowIds(flatRows), [flatRows])
  const persistSessionOrder = useCallback(
    (ids: string[]) => onReorderSessions?.(mergeVisibleReorder(allSortableRowIds, ids)),
    [allSortableRowIds, onReorderSessions]
  )

  useEffect(() => {
    if (grouping !== 'date' && grouping !== 'status') {
      return
    }

    $sidebarListGroupIds.set(
      flatRows.flatMap(row => (row.kind === 'divider' ? [listGroupNodeId(row.key)] : []))
    )

    return () => {
      $sidebarListGroupIds.set([])
    }
  }, [flatRows, grouping])

  // Pinned never virtualizes. Virtualization needs a bounded viewport to
  // measure against, and Pinned deliberately has none — however many chats you
  // pin, all of them render and the sidebar's own scroll carries the length.
  const flatVirtualized =
    !pinned &&
    !showEmptyState &&
    !groups?.length &&
    !projectOverview?.length &&
    !projectContent &&
    sessions.length >= VIRTUALIZE_THRESHOLD

  // First paint into the grouped view (e.g. the app restoring the Projects tab)
  // has flat recents in `sessions` but no tree yet. Show skeletons rather than
  // flashing the flat session list until the overview/content/groups resolve. A
  // background refresh keeps the prior tree, so this only fires when empty.
  const showProjectsSkeleton =
    projectsLoading && !hasProjectOverview && !hasProjectContent && !projectContent && !groups?.length

  let inner: React.ReactNode

  if (showProjectsSkeleton) {
    inner = <SidebarSessionSkeletons />
  } else if (projectContent) {
    // Entered a project: the back row is always present, then either the
    // (overlay-aware) content or a clean empty state — never a bare spinner or a
    // blank pane while lanes hydrate.
    inner = (
      <>
        {projectBackRow}
        {hasProjectContent ? (
          <EnteredProjectContent
            liveSessions={liveSessions}
            onNewSession={onNewSessionInWorkspace}
            project={projectContent}
            removedSessionIds={removedSessionIds}
            renderRows={renderRowsDated}
            repoWorktrees={projectRepoWorktrees}
          />
        ) : (
          emptyState
        )}
      </>
    )
  } else if (showEmptyState) {
    inner = emptyState
  } else if (projectOverview?.length) {
    // The model is already ordered (Home leads; then the default sort groups
    // explicit-before-auto, with a manual drag-order winning when present).
    // Render in that order and make rows drag-to-reorder when a handler is
    // wired — Home stays outside the sortable list, it's a fixture.
    const home = projectOverview[0]?.isNoProject ? projectOverview[0] : undefined
    const sortableProjects = home ? projectOverview.slice(1) : projectOverview
    const projectsDraggable = sortableProjects.length > 1 && !!onReorderProjects
    const Row = projectsDraggable ? SortableProjectOverviewRow : ProjectOverviewRow

    const projectRow = (project: SidebarProjectTree, Component: typeof ProjectOverviewRow) => (
      <Component
        activeProjectId={activeProjectId}
        key={project.id}
        onEnter={onEnterProject}
        onNewSession={onNewSessionInWorkspace}
        previewSessions={projectOverviewPreviews?.[project.id]}
        project={project}
        renderRows={renderRows}
      />
    )

    const rows = sortableProjects.map(project => projectRow(project, Row))

    inner = (
      <>
        {home && projectRow(home, ProjectOverviewRow)}
        {projectsDraggable && onReorderProjects ? (
          <ReorderableList
            ids={sortableProjects.map(project => project.id)}
            onReorder={onReorderProjects}
            sensors={dndSensors}
          >
            {rows}
          </ReorderableList>
        ) : (
          rows
        )}
      </>
    )
  } else if (groups?.length) {
    // Profile/source groups never reorder; render them flat with static rows.
    inner = groups.map(group => (
      <SidebarWorkspaceGroup
        group={group}
        key={group.id}
        onNewSession={onNewSessionInWorkspace}
        renderRows={renderRows}
      />
    ))
  } else if (flatVirtualized) {
    const virtual = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        card={card}
        className={contentClassName}
        dividerAction={dividerAction}
        dividerToggle={dividerToggle}
        onArchiveSession={onArchiveSession}
        onBranchSession={onBranchSession}
        onDeleteSession={onDeleteSession}
        onResumeSession={onResumeSession}
        onTogglePin={onTogglePin}
        onToggleUnread={onToggleUnread}
        pinned={pinned}
        rows={visibleRows}
        showProfileTags={showProfileTags}
        sortable={sessionsDraggable}
      />
    )

    inner =
      sessionsDraggable && onReorderSessions ? (
        <ReorderableList ids={sortableRowIds} onReorder={persistSessionOrder} sensors={dndSensors}>
          {virtual}
        </ReorderableList>
      ) : (
        virtual
      )
  } else if (sessionsDraggable && onReorderSessions) {
    inner = (
      <ReorderableList ids={sortableRowIds} onReorder={persistSessionOrder} sensors={dndSensors}>
        {visibleRows.map(row => renderListRow(row, true, dividerAction))}
      </ReorderableList>
    )
  } else {
    inner = visibleRows.map(row => renderListRow(row, false, dividerAction))
  }

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container. Both axes: `overflow-y-visible` next
  // to the inherited `overflow-x-hidden` computes to `auto` (CSS spec), which
  // kept a phantom 4px scrollbar gutter and cut every row short on the right.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-visible')

  return (
    <SidebarGroup className={rootClassName}>
      <SidebarSectionHeader
        action={headerAction}
        collapsible={collapsible}
        icon={labelIcon}
        label={label}
        meta={labelMeta}
        onToggle={onToggle}
        open={sectionOpen}
      />
      {sectionOpen && (
        <SidebarGroupContent className={resolvedContentClassName}>
          {inner}
          {footer}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

interface SortableSessionRowProps {
  session: SessionInfo
  isPinned: boolean
  isSelected: boolean
  unread: boolean
  onArchive: () => void
  onDelete: () => void
  onPin: () => void
  onToggleUnread: () => void
  onResume: () => void
}

function SortableSidebarSessionRow(props: SortableSessionRowProps) {
  return <SidebarSessionRow {...props} {...useSortableBindings(props.session.id)} />
}

function SortableProjectOverviewRow(props: React.ComponentProps<typeof ProjectOverviewRow>) {
  return <ProjectOverviewRow {...props} {...useSortableBindings(props.project.id)} />
}
