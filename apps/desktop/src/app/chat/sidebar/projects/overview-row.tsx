import type * as React from 'react'
import { useRef } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import {
  SIDEBAR_LEAD_ICON_SIZE,
  SidebarCount,
  SidebarRowBody,
  SidebarRowCluster,
  SidebarRowGrab,
  SidebarRowLabel,
  SidebarRowLead,
  SidebarRowLeadGlyph,
  SidebarRowLink,
  SidebarRowNest,
  SidebarRowShell
} from '../chrome'

import { latestProjectSessions, PROJECT_PREVIEW_COUNT, useWorkspaceNodeOpen } from './model'
import { ProjectMenu } from './project-menu'
import type { SidebarProjectTree } from './workspace-groups'
import { WorkspaceAddButton } from './workspace-header'

// A bare color dot (no icon) or an icon glyph — tinted by `color` when set, else
// the lead's default tertiary. The glyph wrapper centers + caps size either way.
export function projectIcon({ color, icon }: SidebarProjectTree) {
  if (color && !icon) {
    return (
      <SidebarRowLeadGlyph>
        <span aria-hidden="true" className="size-1 rounded-full" style={{ backgroundColor: color }} />
      </SidebarRowLeadGlyph>
    )
  }

  return (
    <SidebarRowLeadGlyph style={color ? { color } : undefined}>
      <Codicon name={icon || 'folder-library'} size={SIDEBAR_LEAD_ICON_SIZE} />
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
  const projectLabel = project.isNoProject ? s.noProject : project.label
  const [open, toggleOpen] = useWorkspaceNodeOpen(project.id)
  // The appearance popover anchors here (the full row) so it opens flush with
  // the sidebar's content edge regardless of which side the sidebar is on.
  const rowRef = useRef<HTMLDivElement>(null)
  const fetched = (previewSessions ?? []).slice(0, PROJECT_PREVIEW_COUNT)
  const preview = renderRows ? (fetched.length ? fetched : latestProjectSessions(project, PROJECT_PREVIEW_COUNT)) : []
  const remainingCount = Math.max(0, project.sessionCount - preview.length)

  const lead = reorderable ? (
    <SidebarRowGrab
      ariaLabel={s.projects.reorder(projectLabel)}
      dragging={dragging}
      dragHandleProps={dragHandleProps}
      leadClassName="overflow-visible"
    >
      {projectIcon(project)}
    </SidebarRowGrab>
  ) : (
    <SidebarRowLead>{projectIcon(project)}</SidebarRowLead>
  )

  return (
    <div className={cn(dragging && 'relative z-10')} ref={ref} style={style}>
      <SidebarRowShell
        actions={
          <>
            {onNewSession && (
              <WorkspaceAddButton label={s.newSessionIn(projectLabel)} onClick={() => onNewSession(project.path)} />
            )}
            <ProjectMenu anchorRef={rowRef} isActive={isActive} project={project} />
          </>
        }
        className={cn('group/workspace', dragging && 'cursor-grabbing bg-(--ui-sidebar-surface-background)')}
        ref={rowRef}
      >
        <SidebarRowCluster className="min-w-0 flex-1">
          {lead}
          <SidebarRowLink
            aria-label={s.projects.enter(projectLabel)}
            labelClassName={cn('hover:text-foreground hover:underline', isActive && 'text-foreground')}
            onClick={() => onEnter?.(project.id)}
          >
            {projectLabel}
          </SidebarRowLink>
          {project.sessionCount > 0 && <SidebarCount>{project.sessionCount}</SidebarCount>}
          {preview.length > 0 ? (
            <button
              aria-label={s.projects.toggle(projectLabel)}
              className="flex flex-1 items-center self-stretch bg-transparent p-0"
              onClick={toggleOpen}
              type="button"
            >
              <DisclosureCaret
                className="shrink-0 text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
                open={open}
              />
            </button>
          ) : (
            <span className="flex-1" />
          )}
        </SidebarRowCluster>
      </SidebarRowShell>
      {open && preview.length > 0 && (
        <SidebarRowNest>
          {renderRows?.(preview)}
          {remainingCount > 0 && onEnter && (
            <SidebarRowShell>
              <SidebarRowBody
                className="group/show-more text-(--ui-text-tertiary) hover:text-foreground"
                onClick={() => onEnter(project.id)}
              >
                <SidebarRowLead>
                  <SidebarRowLeadGlyph>
                    <Codicon name="ellipsis" size={SIDEBAR_LEAD_ICON_SIZE} />
                  </SidebarRowLeadGlyph>
                </SidebarRowLead>
                <SidebarRowLabel className="text-xs underline-offset-4 group-hover/show-more:underline">
                  {s.showMoreIn(remainingCount, projectLabel)}
                </SidebarRowLabel>
              </SidebarRowBody>
            </SidebarRowShell>
          )}
        </SidebarRowNest>
      )}
    </div>
  )
}
