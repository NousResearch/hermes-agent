import type { useSensors } from '@dnd-kit/core'
import type * as React from 'react'
import { useMemo } from 'react'

import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { flattenSessionsWithBranches } from '@/lib/session-branch-tree'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'

import { SidebarCount } from './chrome'
import {
  EnteredProjectContent,
  flattenProjectSessions,
  ProjectOverviewRow,
  type ProjectSessionSort,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  SidebarWorkspaceGroup,
  type SidebarWorkspaceTree
} from './projects'
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
      {meta && <SidebarCount>{meta}</SidebarCount>}
    </>
  )

  return (
    <div className="group/section flex shrink-0 items-center justify-between gap-1 pb-1 pt-1.5">
      {collapsible ? (
        <button
          className="group/section-label flex w-fit items-center gap-1 bg-transparent text-left leading-none"
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
        <div className="flex w-fit items-center gap-1 leading-none">{labelBody}</div>
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
  workingSessionIdSet: Set<string>
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onBranchSession?: (sessionId: string, profile?: string) => void
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
  tree?: SidebarWorkspaceTree[]
  // Project overview: when present, render a drill-in list of project rows
  // instead of sessions. Clicking a row enters that project (onEnterProject),
  // which then passes `projectContent` on the next render. Takes precedence
  // over `tree` / `groups`.
  projectOverview?: SidebarProjectTree[]
  // Per-project preview rows (from the backend tree), keyed by project path.
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
  // Sort preference for session rows inside the Project overview and drill-in tree.
  projectSessionLocale?: string
  projectSessionSort?: ProjectSessionSort
  activeProjectId?: null | string
  labelMeta?: React.ReactNode
  labelIcon?: React.ReactNode
  // When false the section header is static (no caret/toggle) and always open.
  collapsible?: boolean
  sortable?: boolean
  // The flat session list is the only hand-reorderable surface (grouped/project
  // views sort deterministically), so it owns the one ReorderableList.
  onReorderSessions?: (ids: string[]) => void
  // Drag-to-reorder for the project overview list (top-level projects).
  onReorderProjects?: (ids: string[]) => void
  // Rendered atop the entered-project body (a "back to overview" row).
  projectBackRow?: React.ReactNode
  dndSensors?: ReturnType<typeof useSensors>
}

export function SidebarSessionsSection({
  label,
  open,
  onToggle,
  sessions,
  activeSessionId,
  workingSessionIdSet,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onBranchSession,
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
  projectOverview,
  projectOverviewPreviews,
  projectsLoading = false,
  onEnterProject,
  projectContent,
  projectRepoWorktrees,
  liveSessions,
  removedSessionIds,
  projectSessionLocale,
  projectSessionSort,
  activeProjectId,
  labelMeta,
  labelIcon,
  collapsible = true,
  sortable = false,
  onReorderSessions,
  onReorderProjects,
  projectBackRow,
  dndSensors
}: SidebarSessionsSectionProps) {
  const sectionOpen = collapsible ? open : true
  const hasGroupedSessions = Boolean(groups?.some(group => group.sessions.length > 0))
  // A defined project list is itself content (even an empty project should
  // render as a drill-in row so the user can see it exists).
  const hasProjectOverview = Boolean(projectOverview?.length)
  const hasProjectContent = Boolean(projectContent && projectContent.sessionCount > 0)

  const showEmptyState =
    forceEmptyState || (!hasGroupedSessions && !hasProjectOverview && !hasProjectContent && sessions.length === 0)

  // The flat recents/pinned list is the only place sessions reorder by hand;
  // grouped/tree views always sort by creation date and never drag.
  const sessionsDraggable = sortable && !!onReorderSessions
  const displayEntries = useMemo(() => flattenSessionsWithBranches(sessions), [sessions])

  const renderRow = (session: SessionInfo, draggable: boolean, branchStem?: string) => {
    const rowProps = {
      branchStem,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onBranch: onBranchSession ? () => onBranchSession(session.id, session.profile) : undefined,
      onDelete: () => onDeleteSession(session.id),
      onPin: () => onTogglePin(sessionPinId(session)),
      onResume: () => onResumeSession(session.id),
      reorderable: draggable && !branchStem,
      session
    }

    return draggable && !branchStem ? (
      <SortableSidebarSessionRow key={session.id} {...rowProps} />
    ) : (
      <SidebarSessionRow key={session.id} {...rowProps} />
    )
  }

  // Project rows are static and can opt into a persisted title sort; every other
  // sidebar surface keeps its own existing order and drag semantics.
  const renderRows = (items: SessionInfo[]) =>
    (projectSessionSort
      ? flattenProjectSessions(items, projectSessionSort, projectSessionLocale)
      : flattenSessionsWithBranches(items)
    ).map(
      ({ branchStem, session }) => renderRow(session, false, branchStem)
    )

  const flatVirtualized =
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
            renderRows={renderRows}
            repoWorktrees={projectRepoWorktrees}
            sessionLocale={projectSessionLocale}
            sessionSort={projectSessionSort ?? 'recent'}
          />
        ) : (
          emptyState
        )}
      </>
    )
  } else if (showEmptyState) {
    inner = emptyState
  } else if (projectOverview?.length) {
    // The model is already ordered (default sort groups explicit-before-auto;
    // a manual drag-order, when present, wins). Render in that order and make
    // rows drag-to-reorder when a handler is wired.
    const projectsDraggable = projectOverview.length > 1 && !!onReorderProjects
    const Row = projectsDraggable ? SortableProjectOverviewRow : ProjectOverviewRow

    const rows = projectOverview.map(project => (
      <Row
        activeProjectId={activeProjectId}
        key={project.id}
        onEnter={onEnterProject}
        onNewSession={onNewSessionInWorkspace}
        previewSessions={project.path ? projectOverviewPreviews?.[project.path] : undefined}
        project={project}
        renderRows={renderRows}
      />
    ))

    inner =
      projectsDraggable && onReorderProjects ? (
        <ReorderableList
          ids={projectOverview.map(project => project.id)}
          onReorder={onReorderProjects}
          sensors={dndSensors}
        >
          {rows}
        </ReorderableList>
      ) : (
        rows
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
        className={contentClassName}
        entries={displayEntries}
        onArchiveSession={onArchiveSession}
        onBranchSession={onBranchSession}
        onDeleteSession={onDeleteSession}
        onResumeSession={onResumeSession}
        onTogglePin={onTogglePin}
        pinned={pinned}
        sortable={sessionsDraggable}
        workingSessionIdSet={workingSessionIdSet}
      />
    )

    inner =
      sessionsDraggable && onReorderSessions ? (
        <ReorderableList ids={sessions.map(s => s.id)} onReorder={onReorderSessions} sensors={dndSensors}>
          {virtual}
        </ReorderableList>
      ) : (
        virtual
      )
  } else if (sessionsDraggable && onReorderSessions) {
    inner = (
      <ReorderableList ids={sessions.map(s => s.id)} onReorder={onReorderSessions} sensors={dndSensors}>
        {displayEntries.map(({ branchStem, session }) => renderRow(session, true, branchStem))}
      </ReorderableList>
    )
  } else {
    inner = displayEntries.map(({ branchStem, session }) => renderRow(session, false, branchStem))
  }

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

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
  isWorking: boolean
  onArchive: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
}

function SortableSidebarSessionRow(props: SortableSessionRowProps) {
  return <SidebarSessionRow {...props} {...useSortableBindings(props.session.id)} />
}

function SortableProjectOverviewRow(props: React.ComponentProps<typeof ProjectOverviewRow>) {
  return <ProjectOverviewRow {...props} {...useSortableBindings(props.project.id)} />
}
