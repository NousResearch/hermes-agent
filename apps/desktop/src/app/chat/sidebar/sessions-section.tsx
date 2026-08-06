import type { useSensors } from '@dnd-kit/core'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { ActionsMenu, renderActionItem } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { flattenSessionsWithBranches } from '@/lib/session-branch-tree'
import {
  groupEntriesByRecency,
  sessionDateGroupKeys,
  type SidebarListRow,
  toSessionRows,
  visibleSessionDateGroupRows
} from '@/lib/session-date-groups'
import { localeWeekStartDay, sessionBucketLabel } from '@/lib/time'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'
import {
  $collapsedSessionDateGroups,
  collapseAllSessionDateGroups,
  expandAllSessionDateGroups,
  setSessionDateGroupCollapsed
} from '@/store/session-date-group-collapse'

import { SidebarDateDivider, SidebarSectionMeta } from './chrome'
import {
  EnteredProjectContent,
  ProjectOverviewRow,
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
  // Insert calendar date disclosures into chronological lists.
  dateGrouped?: boolean
  // Renderer-only persistence namespace (backend connection + active profile).
  dateGroupScope?: string
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
  activeProjectId,
  labelMeta,
  labelIcon,
  collapsible = true,
  sortable = false,
  onReorderSessions,
  onReorderProjects,
  projectBackRow,
  dndSensors,
  showProfileTags = false,
  dateGrouped = false,
  dateGroupScope
}: SidebarSessionsSectionProps) {
  const { locale, t } = useI18n()
  const collapsedGroupsByScope = useStore($collapsedSessionDateGroups)
  const dividerLabels = t.sidebar.dateDivider

  const collapsedGroupKeys = useMemo(
    () => new Set(dateGroupScope ? (collapsedGroupsByScope[dateGroupScope] ?? []) : []),
    [collapsedGroupsByScope, dateGroupScope]
  )

  const weekStartsOn = localeWeekStartDay(locale)
  const [groupingNowMs, setGroupingNowMs] = useState(() => Date.now())

  useEffect(() => {
    if (!dateGrouped) {
      return
    }

    let timeoutId: number

    const scheduleNextLocalMidnight = () => {
      const now = Date.now()
      const nextMidnight = new Date(now)

      nextMidnight.setHours(24, 0, 0, 0)
      timeoutId = window.setTimeout(() => {
        setGroupingNowMs(Date.now())
        scheduleNextLocalMidnight()
      }, nextMidnight.getTime() - now)
    }

    scheduleNextLocalMidnight()

    return () => window.clearTimeout(timeoutId)
  }, [dateGrouped])

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

  // First paint into the grouped view (e.g. the app restoring the Projects tab)
  // has flat recents in `sessions` but no tree yet. Show skeletons rather than
  // flashing the flat session list until the overview/content/groups resolve. A
  // background refresh keeps the prior tree, so this only fires when empty.
  const showProjectsSkeleton =
    projectsLoading && !hasProjectOverview && !hasProjectContent && !projectContent && !groups?.length

  // The flat recents/pinned list is the only place sessions reorder by hand;
  // grouped/tree views always sort by creation date and never drag.
  const sessionsDraggable = sortable && !!onReorderSessions
  // Pinned and manual-drag lists pass sessions already in the caller's order.
  // Default recents stay dateGrouped and still re-sort roots by group recency
  // so partition buckets stay truthful — but NEVER for pins, where a turn
  // finishing was floating background tasks over the user's fixed ranking.
  const preserveInputOrder = pinned || (sessionsDraggable && !dateGrouped)

  const displayEntries = useMemo(
    () => flattenSessionsWithBranches(sessions, { preserveOrder: preserveInputOrder }),
    [sessions, preserveInputOrder]
  )

  const flatRows: SidebarListRow[] = useMemo(
    () =>
      dateGrouped
        ? groupEntriesByRecency(displayEntries, groupingNowMs, weekStartsOn)
        : toSessionRows(displayEntries),
    [dateGrouped, displayEntries, groupingNowMs, weekStartsOn]
  )

  const visibleFlatRows = useMemo(
    () => visibleSessionDateGroupRows(flatRows, collapsedGroupKeys),
    [collapsedGroupKeys, flatRows]
  )

  // The exact branch the `inner` if/else below will pick — mirrored here (not
  // re-derived from prop presence) so key aggregation only counts a temporal
  // list when it is the one that actually renders.
  const activeRenderBranch = showProjectsSkeleton
    ? 'skeleton'
    : projectContent
      ? 'projectContent'
      : showEmptyState
        ? 'empty'
        : projectOverview?.length
          ? 'projectOverview'
          : groups?.length
            ? 'groups'
            : 'flat'

  // Entered-project lanes and profile/source groups each run their own
  // `renderRowsDated` grouping pass, but a bucket key depends only on an
  // entry's own recency — never on which lane/group it lands in — so pooling
  // every rendered list's sessions into one grouping pass yields the same key
  // set as unioning each list's own pass separately.
  const dateGroupKeysFor = useCallback(
    (sessionsToGroup: SessionInfo[]): string[] =>
      dateGrouped && sessionsToGroup.length
        ? sessionDateGroupKeys(
            groupEntriesByRecency(flattenSessionsWithBranches(sessionsToGroup), groupingNowMs, weekStartsOn)
          )
        : [],
    [dateGrouped, groupingNowMs, weekStartsOn]
  )

  const knownDateGroupKeys = useMemo(
    () => [
      ...new Set([
        ...(activeRenderBranch === 'flat' ? sessionDateGroupKeys(flatRows) : []),
        ...(activeRenderBranch === 'projectContent' && hasProjectContent && projectContent
          ? dateGroupKeysFor(projectContent.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions)))
          : []),
        ...(activeRenderBranch === 'groups' && groups ? dateGroupKeysFor(groups.flatMap(group => group.sessions)) : [])
      ])
    ],
    [activeRenderBranch, dateGroupKeysFor, flatRows, groups, hasProjectContent, projectContent]
  )

  const toggleDateGroup = useCallback(
    (key: string) => {
      if (!dateGroupScope) {
        return
      }

      setSessionDateGroupCollapsed(dateGroupScope, key, !collapsedGroupKeys.has(key))
    },
    [collapsedGroupKeys, dateGroupScope]
  )

  const renderRow = useCallback(
    (session: SessionInfo, draggable: boolean, branchStem?: string) => {
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
        session,
        showProfile: showProfileTags
      }

      return draggable && !branchStem ? (
        <SortableSidebarSessionRow key={session.id} {...rowProps} />
      ) : (
        <SidebarSessionRow key={session.id} {...rowProps} />
      )
    },
    [
      activeSessionId,
      onArchiveSession,
      onBranchSession,
      onDeleteSession,
      onResumeSession,
      onTogglePin,
      pinned,
      showProfileTags,
      workingSessionIdSet
    ]
  )

  // A single flat/virtual/lane list row — either a date disclosure or a session.
  const renderListRow = useCallback(
    (row: SidebarListRow, draggable: boolean) => {
      if (row.kind === 'session') {
        return renderRow(row.entry.session, draggable, row.entry.branchStem)
      }

      const label = sessionBucketLabel(row.bucket, dividerLabels, locale, groupingNowMs)
      const expanded = !collapsedGroupKeys.has(row.key)

      return (
        <SidebarDateDivider
          aria-label={expanded ? dividerLabels.collapseGroup(label) : dividerLabels.expandGroup(label)}
          key={row.rowKey}
          label={label}
          onToggle={() => toggleDateGroup(row.key)}
          open={expanded}
        />
      )
    },
    [collapsedGroupKeys, dividerLabels, groupingNowMs, locale, renderRow, toggleDateGroup]
  )

  // Project overview previews are a short, date-ordered static peek — no
  // dividers, the overview list itself isn't chronological.
  const renderRows = useCallback(
    (items: SessionInfo[]) =>
      flattenSessionsWithBranches(items).map(({ branchStem, session }) => renderRow(session, false, branchStem)),
    [renderRow]
  )

  // Same as `renderRows`, but with date dividers folded in — used for
  // entered-project lanes and profile/source groups so a list spanning
  // multiple days reads chronologically, matching the flat recents list.
  const renderRowsDated = useCallback(
    (items: SessionInfo[]) => {
      const entries = flattenSessionsWithBranches(items)

      const rows = dateGrouped
        ? groupEntriesByRecency(entries, groupingNowMs, weekStartsOn)
        : toSessionRows(entries)

      return visibleSessionDateGroupRows(rows, collapsedGroupKeys).map(row => renderListRow(row, false))
    },
    [collapsedGroupKeys, dateGrouped, groupingNowMs, renderListRow, weekStartsOn]
  )

  const flatVirtualized =
    !showEmptyState &&
    !groups?.length &&
    !projectOverview?.length &&
    !projectContent &&
    sessions.length >= VIRTUALIZE_THRESHOLD

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
    // Profile/source groups never reorder, but each group's own sessions still
    // read chronologically, so date-group them the same way entered-project
    // lanes do.
    inner = groups.map(group => (
      <SidebarWorkspaceGroup
        group={group}
        key={group.id}
        onNewSession={onNewSessionInWorkspace}
        renderRows={renderRowsDated}
      />
    ))
  } else if (flatVirtualized) {
    const virtual = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        className={contentClassName}
        collapsedDateGroupKeys={collapsedGroupKeys}
        onArchiveSession={onArchiveSession}
        onBranchSession={onBranchSession}
        onDeleteSession={onDeleteSession}
        onResumeSession={onResumeSession}
        onToggleDateGroup={toggleDateGroup}
        onTogglePin={onTogglePin}
        pinned={pinned}
        rows={visibleFlatRows}
        showProfileTags={showProfileTags}
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
        {visibleFlatRows.map(row => renderListRow(row, true))}
      </ReorderableList>
    )
  } else {
    inner = visibleFlatRows.map(row => renderListRow(row, false))
  }

  // Gated on the groups actually known from rendered temporal lists (above),
  // not on prop presence: a projectOverview view, which never groups by date,
  // never forces this menu on.
  const showDateGroupMenu = Boolean(dateGroupScope) && knownDateGroupKeys.length > 0

  const dateGroupMenu = showDateGroupMenu ? (
    <ActionsMenu
      ariaLabel={dividerLabels.actions}
      items={kit => (
        <>
          {renderActionItem(kit, {
            key: 'collapse-all-date-groups',
            label: dividerLabels.collapseAll,
            onSelect: event => {
              event.stopPropagation()

              if (dateGroupScope) {
                collapseAllSessionDateGroups(dateGroupScope, knownDateGroupKeys)
              }
            }
          })}
          {renderActionItem(kit, {
            key: 'expand-all-date-groups',
            label: dividerLabels.expandAll,
            onSelect: event => {
              event.stopPropagation()

              if (dateGroupScope) {
                expandAllSessionDateGroups(dateGroupScope, knownDateGroupKeys)
              }
            }
          })}
        </>
      )}
    >
      <Button
        aria-label={dividerLabels.actions}
        className="size-5 rounded-[4px] bg-transparent text-(--ui-text-tertiary) hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:ring-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground"
        size="icon"
        variant="ghost"
      >
        <Codicon name="kebab-vertical" size="0.75rem" />
      </Button>
    </ActionsMenu>
  ) : null

  const resolvedHeaderAction =
    headerAction || dateGroupMenu ? (
      <div className="flex shrink-0 items-center gap-0.5">
        {headerAction}
        {dateGroupMenu}
      </div>
    ) : undefined

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

  return (
    <SidebarGroup className={rootClassName}>
      <SidebarSectionHeader
        action={resolvedHeaderAction}
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
