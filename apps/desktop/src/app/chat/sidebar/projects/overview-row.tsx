import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useRef, useState } from 'react'

import { type NewSessionSplitHandler, startNewSessionDrag } from '@/app/chat/new-session-drag'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $sidebarShowAllSessions } from '@/store/layout'
import { $projectScope } from '@/store/project-scope'
import { $projectTree, enterProject, fetchProjectSessions, projectProfile } from '@/store/projects'
import type { SessionDotState } from '@/store/session-dot-state'

import { sessionDotClassName, sessionDotLabel } from '../../session-status-dot'
import {
  SIDEBAR_LEAD_ICON_SIZE,
  SidebarGroupRow,
  SidebarRowBody,
  SidebarRowGrab,
  SidebarRowLabel,
  SidebarRowLead,
  SidebarRowLeadGlyph,
  SidebarRowLink,
  SidebarRowNest,
  SidebarRowShell
} from '../chrome'
import { shellOwnsPress } from '../reorderable-list'

import {
  expandedProjectSessions,
  latestProjectSessions,
  PROJECT_PREVIEW_COUNT,
  PROJECT_SESSION_PAGE,
  projectBackTarget,
  useRevealedRows,
  useWorkspaceNodeOpen
} from './model'
import { ProjectContextMenu, ProjectMenu } from './project-menu'
import { excludeProjectSessions, type SidebarProjectTree } from './workspace-groups'
import { WorkspaceAddButton, WorkspaceShowMoreRow } from './workspace-header'

// A bare color dot (no icon) or an icon glyph — tinted by `color` when set, else
// the lead's default tertiary. The glyph wrapper centers + caps size either way.
// Auto-discovered repos (git lanes Desktop found by scanning disk, not rows in
// projects.db) get the `repo` glyph so a glance tells explicit projects
// (`folder-library`) apart from incidental disk/session findings.
export function projectIcon({ color, icon, isAuto, isNoProject }: SidebarProjectTree) {
  if (color && !icon) {
    return (
      <SidebarRowLeadGlyph>
        <span aria-hidden="true" className="size-1 rounded-full" style={{ backgroundColor: color }} />
      </SidebarRowLeadGlyph>
    )
  }

  return (
    <SidebarRowLeadGlyph style={color ? { color } : undefined}>
      <Codicon
        name={icon || (isNoProject ? 'home' : isAuto ? 'repo' : 'folder-library')}
        size={SIDEBAR_LEAD_ICON_SIZE}
      />
    </SidebarRowLeadGlyph>
  )
}

/**
 * The entered view's back row. Stepping back walks OUT one level at a time: a project that nests
 * under another was entered from it, so the arrow lands on that parent — and only a top-level
 * project (or a parent the tree no longer carries) falls through to `onExit`, the caller's own move
 * back to the overview.
 */
export function ProjectBackRow({ label, onExit }: { label: string; onExit: () => void }) {
  const projects = useStore($projectTree)
  const scope = useStore($projectScope)
  const parent = projectBackTarget(projects, scope)

  return (
    <SidebarRowShell>
      <SidebarRowBody
        className="group/back w-full text-(--ui-text-tertiary) opacity-40 hover:text-foreground"
        onClick={() => (parent ? enterProject(parent) : onExit())}
      >
        <SidebarRowLead>
          <SidebarRowLeadGlyph>
            <Codicon name="arrow-left" size={SIDEBAR_LEAD_ICON_SIZE} />
          </SidebarRowLeadGlyph>
        </SidebarRowLead>
        <SidebarRowLabel className="text-xs underline-offset-4 group-hover/back:underline">{label}</SidebarRowLabel>
      </SidebarRowBody>
    </SidebarRowShell>
  )
}

interface ProjectOverviewRowProps {
  project: SidebarProjectTree
  onEnter?: (id: string) => void
  onNewSession?: (path: null | string) => void
  /** Drag the project's "+" onto a chat zone: create a new session pinned to
   *  this project's cwd, placed exactly where it's dropped. */
  onNewSessionSplit?: NewSessionSplitHandler
  renderRows?: (sessions: SessionInfo[]) => React.ReactNode
  activeProjectId?: null | string
  previewSessions?: SessionInfo[]
  /** What the project tree drops (pins, filter misses, just-deleted rows) —
   *  the same predicate `previewSessions` was built with, so a "Show all"
   *  hydration can't resurrect them. */
  isSessionHidden?: (session: SessionInfo) => boolean
  /** How many of the backend's `sessionCount` that predicate hides, so
   *  "Show all N" promises only rows the view will actually render. */
  hiddenSessionCount?: number
  /** The loudest status anywhere in this project's subtree — its own sessions and every nested
   *  project's. Painted only while the row is collapsed: an expanded row shows its sessions' own
   *  dots, and a second dot on the parent would just double the ink. */
  attentionState?: SessionDotState
  /** Whether any project nests under this one. A subproject-only parent has no preview rows to fold,
   *  but it still has children — which is exactly what the fold is for. */
  hasNestedProjects?: boolean
  /** How many levels in this row sits (0 = top level). The overview passes the project's real depth,
   *  so a subproject of a subproject is indented under it rather than beside it; the entered view
   *  passes 1, because the rows it draws are all one level inside the project you are in. */
  depth?: number
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  ref?: React.Ref<HTMLDivElement>
  style?: React.CSSProperties
}

// #124808: a path-less explicit project (multi-folder, never assigned a
// primary_path) still carries repo roots. Its trunk "+" must anchor at
// the first repo root — passing the null wire path through would take the
// reserved Home/detached branch downstream and silently create a global
// session. Home itself keeps null ("no folder" is its contract).
function overviewRowNewSessionPath(project: ProjectOverviewRowProps['project']): null | string {
  if (project.isNoProject || (project.path ?? '').trim()) {
    return project.path
  }

  return (project.repos ?? []).map(repo => repo.path).find(root => (root ?? '').trim()) ?? project.path
}

/** A project row's label cell: the enter link (an auto-discovered project names its cue for screen
 *  readers, which the aria-hidden glyph can't) and the subtree's status dot. */
function ProjectRowLabel({
  attention,
  attentionLabel,
  isActive,
  onEnter,
  project
}: Pick<ProjectOverviewRowProps, 'onEnter' | 'project'> & {
  attention: null | Parameters<typeof sessionDotLabel>[0]
  attentionLabel: null | ReturnType<typeof sessionDotLabel>
  isActive: boolean
}) {
  const { t } = useI18n()
  const s = t.sidebar

  const labelLink = (
    <SidebarRowLink
      // The glyph is aria-hidden and the tooltip only speaks on hover, so the
      // link's own name carries the auto cue — screen readers get it too.
      aria-label={
        project.isAuto
          ? `${s.projects.enter(project.label)} (${s.projects.autoDiscovered})`
          : s.projects.enter(project.label)
      }
      labelClassName={cn('hover:text-foreground hover:underline', isActive && 'text-foreground')}
      onClick={() => onEnter?.(project.id)}
    >
      {project.label}
    </SidebarRowLink>
  )

  return (
    <>
      {project.isAuto ? <Tip label={s.projects.autoDiscovered}>{labelLink}</Tip> : labelLink}
      {attention && (
        // Same geometry as a session row's dot: a fixed cell, self-centred in the row and
        // centring the dot in itself, so the two dots sit on one axis.
        <span
          aria-label={attentionLabel?.ariaLabel}
          className="grid size-3.5 shrink-0 self-center place-items-center"
          data-project-attention=""
          role="status"
          title={attentionLabel?.title}
        >
          <span className={sessionDotClassName(attention)} />
        </span>
      )}
    </>
  )
}

/** A project row's actions: its ⋯ menu (every real project) and the trunk "+" — the
 *  new-session control, draggable onto a chat zone when the sidebar offers splits. */
function ProjectRowActions({
  isActive,
  newSessionPath,
  onNewSession,
  onNewSessionSplit,
  project,
  rowRef
}: Pick<ProjectOverviewRowProps, 'onNewSession' | 'onNewSessionSplit' | 'project'> & {
  isActive: boolean
  newSessionPath: null | string
  rowRef: React.ComponentProps<typeof ProjectMenu>['anchorRef']
}) {
  const { t } = useI18n()
  const s = t.sidebar

  return (
    <>
      {/* Home is a bucket, not a record, so there's nothing to rename or
          delete — but it still starts sessions: a null path is the "no
          folder" chat. New session sits outermost: it's the one you reach
          for. */}
      {!project.isNoProject && <ProjectMenu anchorRef={rowRef} isActive={isActive} project={project} />}
      {onNewSession && (
        <WorkspaceAddButton
          label={s.newSessionIn(project.label)}
          onClick={() => onNewSession(newSessionPath)}
          onPointerDown={
            onNewSessionSplit
              ? event => {
                  // Drag the "+" onto a chat zone: create the session
                  // pinned to this project's cwd, exactly where it's
                  // dropped. A sub-threshold release falls through to the
                  // onClick above (ordinary new session in main).
                  startNewSessionDrag(
                    placement => {
                      onNewSessionSplit(placement.dir, {
                        anchor: placement.anchor,
                        before: placement.before,
                        cwd: newSessionPath
                      })
                    },
                    event,
                    { cwd: newSessionPath, label: s.newSessionIn(project.label) }
                  )
                }
              : undefined
          }
        />
      )}
    </>
  )
}

export function ProjectOverviewRow({
  project,
  onEnter,
  onNewSession,
  onNewSessionSplit,
  renderRows,
  activeProjectId,
  previewSessions,
  isSessionHidden,
  hiddenSessionCount = 0,
  attentionState,
  hasNestedProjects = false,
  depth,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  ref,
  style
}: ProjectOverviewRowProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const isActive = project.id === activeProjectId
  // One step (0.5rem) further in per nesting level — grouping only, the sessions stay the child's own.
  // The overview passes the row's real depth so a subproject of a subproject is indented under it; a
  // row rendered without one (a lone preview, a test) still indents for its `parentId`, so the indent
  // cannot silently disappear.
  const nestDepth = depth ?? (project.parentId ? 1 : 0)
  const [open, toggleOpen] = useWorkspaceNodeOpen(project.id)
  // The subtree's loudest status, shown whether the row is open or closed: a session that wants an
  // answer inside a subproject must be visible without folding anything open.
  const attention = attentionState && attentionState !== 'idle' ? attentionState : null
  const attentionLabel = attention ? sessionDotLabel(attention, s.row) : null
  // The appearance popover anchors here (the full row) so it opens flush with
  // the sidebar's content edge regardless of which side the sidebar is on.
  const rowRef = useRef<HTMLDivElement>(null)
  const showAllSessions = useStore($sidebarShowAllSessions)
  // The tree payload previews only the most-recent few sessions per project
  // (kept light on purpose); "Show all" hydrates THIS project's lanes on demand
  // rather than widening every project's preview window.
  const [expanded, setExpanded] = useState<SidebarProjectTree | null>(null)
  const [expanding, setExpanding] = useState(false)
  const limit = showAllSessions || expanded ? Infinity : PROJECT_PREVIEW_COUNT
  const fetched = (previewSessions ?? []).slice(0, limit)
  const recent = fetched.length ? fetched : latestProjectSessions(project, limit)
  // The hydrated lanes come straight from the backend, so — like the drill-in
  // (index.tsx) — they haven't been through the tree's exclusion filter yet.
  const visible = expanded && isSessionHidden ? excludeProjectSessions(expanded, isSessionHidden) : expanded
  const preview = renderRows ? (visible ? expandedProjectSessions(recent, visible) : recent) : []
  // Once hydrated, the whole project is reachable but mounts a page at a time
  // (a project can hold thousands of chats; the collapsed preview stays 3).
  const page = useRevealedRows(preview, PROJECT_SESSION_PAGE)
  const rows = expanded ? page.shown : preview
  const total = project.sessionCount - hiddenSessionCount
  const hiddenCount = total - preview.length
  const offerShowAll = !showAllSessions && !expanded && preview.length > 0 && hiddenCount > 0

  const newSessionPath = overviewRowNewSessionPath(project)

  const showAll = () => {
    // All-profiles view has no single backend to ask for one project's lanes;
    // drilling in is the reach there.
    if (!projectProfile()) {
      onEnter?.(project.id)

      return
    }

    setExpanding(true)
    fetchProjectSessions(project.id, { supersedable: false })
      .then(tree => void (tree && setExpanded(tree)))
      .catch(() => onEnter?.(project.id))
      .finally(() => setExpanding(false))
  }

  const lead = reorderable ? (
    <SidebarRowGrab
      ariaLabel={s.projects.reorder(project.label)}
      dragging={dragging}
      dragHandleProps={dragHandleProps}
      leadClassName="overflow-visible"
    >
      {projectIcon(project)}
    </SidebarRowGrab>
  ) : (
    <SidebarRowLead>{projectIcon(project)}</SidebarRowLead>
  )

  const shell = (
    <SidebarGroupRow
      actions={<ProjectRowActions
        isActive={isActive}
        newSessionPath={newSessionPath}
        onNewSession={onNewSession}
        onNewSessionSplit={onNewSessionSplit}
        project={project}
        rowRef={rowRef}
      />}
      className={cn(dragging && 'cursor-grabbing bg-(--ui-sidebar-surface-background)')}
      data-glass-opaque={dragging ? '' : undefined}
      label={
        <ProjectRowLabel
          attention={attention}
          attentionLabel={attentionLabel}
          isActive={isActive}
          onEnter={onEnter}
          project={project}
        />
      }
      lead={lead}
      // The label is grab surface too, not just the lead's grabber — the
      // pointer activator only (the full handle stays on the grabber, see
      // useSortableBindings), minus the controls that keep their own gestures.
      // A project row has no rival drag (its title navigates on CLICK), so the
      // sortable owns the press outright.
      onPointerDown={event => {
        // The project row's ⋯ menu and its confirm dialog portal out of this
        // row's React subtree — a press on either arrives with a target outside
        // the row, so gate the shell on presses that started inside it.
        if (!shellOwnsPress(event)) {
          return
        }

        if ((event.target as HTMLElement).closest('[data-reorder-handle], [data-row-actions]')) {
          return
        }

        dragHandleProps?.onPointerDown?.(event)
      }}
      ref={rowRef}
      toggle={
        preview.length > 0 || hasNestedProjects
          ? { ariaLabel: s.projects.toggle(project.label, !open), onToggle: toggleOpen, open }
          : undefined
      }
      totals={{ costUsd: project.totalCostUsd ?? 0, tokens: project.totalTokens ?? 0 }}
    />
  )

  return (
    // Tag each project sibling with its id so a custom skin can target one
    // project in the overview — the parallel to the entered-project wrapper's
    // `data-sessions-project` (index.tsx), which only fires once you've drilled
    // in. Here it's present on every row of the list.
    <div
      className={cn(
        dragging && 'relative z-10',
        // Painted imperatively by session-drag.ts while a dragged session
        // hovers this row — a live "drop here to move" cue, not React state
        // (it must not repaint the sidebar on every pixel of pointer travel).
        'rounded-[6px] data-[session-drop-hover=true]:outline-2 data-[session-drop-hover=true]:-outline-offset-2 data-[session-drop-hover=true]:outline-sidebar-ring'
      )}
      data-sessions-project={project.id}
      ref={ref}
      // The nesting indent (see `nestDepth`); the virtualizer's own positional styles win where they
      // overlap, so this only ever adds the left padding.
      style={nestDepth ? { ...style, paddingLeft: `${nestDepth * 0.5}rem` } : style}
    >
      {/* Home has no per-project actions, so it gets no right-click menu. */}
      {project.isNoProject ? (
        shell
      ) : (
        <ProjectContextMenu isActive={isActive} project={project}>
          {shell}
        </ProjectContextMenu>
      )}
      {open && preview.length > 0 && (
        <SidebarRowNest>
          {renderRows?.(rows)}
          {offerShowAll && (
            <WorkspaceShowMoreRow disabled={expanding} label={s.projects.showAllCount(total)} onClick={showAll} />
          )}
          {expanded && page.more > 0 && (
            <WorkspaceShowMoreRow label={s.showMoreIn(page.more, project.label)} onClick={page.showMore} />
          )}
        </SidebarRowNest>
      )}
    </div>
  )
}
