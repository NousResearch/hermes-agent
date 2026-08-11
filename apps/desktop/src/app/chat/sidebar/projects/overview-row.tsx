import type * as React from 'react'
import { useRef } from 'react'

import { Codicon } from '@/components/ui/codicon'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

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

import { latestProjectSessions, PROJECT_PREVIEW_COUNT, useWorkspaceNodeOpen } from './model'
import { ProjectContextMenu, ProjectMenu } from './project-menu'
import type { SidebarProjectTree } from './workspace-groups'
import { WorkspaceAddButton } from './workspace-header'

// A bare color dot (no icon) or an icon glyph — tinted by `color` when set, else
// the lead's default tertiary. The glyph wrapper centers + caps size either way.
export function projectIcon({ color, icon, isNoProject }: SidebarProjectTree) {
  if (color && !icon) {
    return (
      <SidebarRowLeadGlyph>
        <span aria-hidden="true" className="size-1 rounded-full" style={{ backgroundColor: color }} />
      </SidebarRowLeadGlyph>
    )
  }

  return (
    <SidebarRowLeadGlyph style={color ? { color } : undefined}>
      <Codicon name={icon || (isNoProject ? 'home' : 'folder-library')} size={SIDEBAR_LEAD_ICON_SIZE} />
    </SidebarRowLeadGlyph>
  )
}

export function ProjectBackRow({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <SidebarRowShell>
      <SidebarRowBody
        className="group/back w-full text-(--ui-text-tertiary) opacity-40 hover:text-foreground"
        onClick={onClick}
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
  renderRows?: (sessions: SessionInfo[]) => React.ReactNode
  activeProjectId?: null | string
  previewSessions?: SessionInfo[]
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  ref?: React.Ref<HTMLDivElement>
  style?: React.CSSProperties
}

export function ProjectOverviewRow({
  project,
  onEnter,
  onNewSession,
  renderRows,
  activeProjectId,
  previewSessions,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  ref,
  style
}: ProjectOverviewRowProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const isActive = project.id === activeProjectId
  const [open, toggleOpen] = useWorkspaceNodeOpen(project.id)
  // The appearance popover anchors here (the full row) so it opens flush with
  // the sidebar's content edge regardless of which side the sidebar is on.
  const rowRef = useRef<HTMLDivElement>(null)
  const fetched = (previewSessions ?? []).slice(0, PROJECT_PREVIEW_COUNT)
  const preview = renderRows ? (fetched.length ? fetched : latestProjectSessions(project, PROJECT_PREVIEW_COUNT)) : []

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
      actions={
        <>
          {/* Home is a bucket, not a record, so there's nothing to rename or
              delete — but it still starts sessions: a null path is the "no
              folder" chat. New session sits outermost: it's the one you reach
              for. */}
          {!project.isNoProject && <ProjectMenu anchorRef={rowRef} isActive={isActive} project={project} />}
          {onNewSession && (
            <WorkspaceAddButton label={s.newSessionIn(project.label)} onClick={() => onNewSession(project.path)} />
          )}
        </>
      }
      className={cn(dragging && 'cursor-grabbing bg-(--ui-sidebar-surface-background)')}
      label={
        <SidebarRowLink
          aria-label={s.projects.enter(project.label)}
          labelClassName={cn('hover:text-foreground hover:underline', isActive && 'text-foreground')}
          onClick={() => onEnter?.(project.id)}
        >
          {project.label}
        </SidebarRowLink>
      }
      lead={lead}
      // The label is grab surface too, not just the lead's grabber — same
      // listeners, minus the controls that keep their own gestures. A project
      // row has no rival drag (its title navigates on CLICK), so the sortable
      // owns the press outright.
      {...dragHandleProps}
      onPointerDown={event => {
        if ((event.target as HTMLElement).closest('[data-reorder-handle], [data-row-actions]')) {
          return
        }

        dragHandleProps?.onPointerDown?.(event)
      }}
      ref={rowRef}
      toggle={
        preview.length > 0
          ? { ariaLabel: s.projects.toggle(project.label, !open), onToggle: toggleOpen, open }
          : undefined
      }
      totals={{ costUsd: project.totalCostUsd ?? 0, tokens: project.totalTokens ?? 0 }}
    />
  )

  return (
    <div className={cn(dragging && 'relative z-10')} ref={ref} style={style}>
      {/* Home has no per-project actions, so it gets no right-click menu. */}
      {project.isNoProject ? (
        shell
      ) : (
        <ProjectContextMenu isActive={isActive} project={project}>
          {shell}
        </ProjectContextMenu>
      )}
      {open && preview.length > 0 && <SidebarRowNest>{renderRows?.(preview)}</SidebarRowNest>}
    </div>
  )
}
