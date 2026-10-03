import type * as React from 'react'

import { useRevealedRows } from '@/app/chat/sidebar/projects/model'
import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorBanner } from '@/components/ui/error-state'
import { RowButton } from '@/components/ui/row-button'
import type { ProjectInfo, SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { fmtDayTime } from '@/lib/time'

import {
  type ActiveProjectSession,
  gitRepositories,
  type LaneKind,
  laneKind,
  type LiveSessionState,
  projectFolders,
  projectPrimaryPath
} from './model'

// Sessions mount a page at a time; a project can hold thousands of chats.
const SESSION_FIRST_PAGE = 20

const STATUS_BADGE: Record<LiveSessionState, 'default' | 'muted' | 'warn'> = {
  background: 'muted',
  'needs-input': 'warn',
  stalled: 'warn',
  working: 'default'
}

export interface ProjectDetailProps {
  active: ActiveProjectSession[]
  info?: ProjectInfo
  onOpenArtifacts: () => void
  onOpenSession: (sessionId: string, event: React.MouseEvent) => void
  onShowInSidebar: () => void
  project: SidebarProjectTree
  recent: SessionInfo[]
  sessionsFailed: boolean
}

export function ProjectDetail({
  active,
  info,
  onOpenArtifacts,
  onOpenSession,
  onShowInSidebar,
  project,
  recent,
  sessionsFailed
}: ProjectDetailProps) {
  const { t } = useI18n()
  const p = t.projects
  const primaryPath = projectPrimaryPath(project, info)
  const folders = projectFolders(project, info)
  const repositories = gitRepositories(project)
  const page = useRevealedRows(recent, SESSION_FIRST_PAGE)

  const laneLabel: Record<LaneKind, string> = {
    kanban: p.laneKanban,
    main: p.laneMain,
    worktree: p.laneWorktree
  }

  return (
    <section aria-label={project.label} className="space-y-6">
      <header className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="min-w-0 truncate text-[0.9375rem] font-semibold tracking-tight">{project.label}</h2>
          {project.isAuto && <Badge variant="muted">{p.autoDiscovered}</Badge>}
        </div>
        {info?.description && (
          <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {info.description}
          </p>
        )}
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-xs">
          <dt className="text-(--ui-text-tertiary)">{p.primaryPath}</dt>
          <dd className="truncate font-mono text-(--ui-text-secondary)">{primaryPath ?? p.noPath}</dd>
          <dt className="text-(--ui-text-tertiary)">{p.sessions}</dt>
          <dd className="text-(--ui-text-secondary)">{p.sessionCount(project.sessionCount)}</dd>
        </dl>
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Button onClick={onOpenArtifacts} size="sm" variant="secondary">
            <Codicon name="files" />
            {p.openArtifacts}
          </Button>
          <Button onClick={onShowInSidebar} size="sm" variant="text">
            {p.showInSidebar}
          </Button>
        </div>
      </header>

      <DetailSection title={p.activeSessions}>
        {active.length === 0 ? (
          <DetailNote>{p.noActiveSessions}</DetailNote>
        ) : (
          <ul className="grid gap-px">
            {active.map(({ session, state }) => (
              <li key={session.id}>
                <SessionRow onOpen={onOpenSession} session={session} status={state} />
              </li>
            ))}
          </ul>
        )}
      </DetailSection>

      <DetailSection title={p.sessions}>
        {sessionsFailed && <ErrorBanner>{p.sessionsFailed}</ErrorBanner>}
        {recent.length === 0 ? (
          active.length === 0 && <DetailNote>{p.noSessions}</DetailNote>
        ) : (
          <ul className="grid gap-px">
            {page.shown.map(session => (
              <li key={session.id}>
                <SessionRow onOpen={onOpenSession} session={session} />
              </li>
            ))}
          </ul>
        )}
        {page.more > 0 && (
          <Button onClick={page.showMore} size="xs" variant="text">
            {t.sidebar.showMoreIn(page.more, project.label)}
          </Button>
        )}
      </DetailSection>

      <DetailSection title={p.repositories}>
        {repositories.length === 0 ? (
          <DetailNote>{p.noRepositories}</DetailNote>
        ) : (
          <ul className="grid gap-3">
            {repositories.map(repo => (
              <li className="space-y-1" key={repo.id}>
                <div className="flex min-w-0 items-baseline gap-2">
                  <Codicon className="shrink-0 self-center text-(--ui-text-tertiary)" name="repo" />
                  <span className="truncate text-xs font-medium">{repo.label}</span>
                  <span className="ml-auto shrink-0 text-[0.65rem] tabular-nums text-(--ui-text-tertiary)">
                    {p.sessionCount(repo.sessionCount)}
                  </span>
                </div>
                {repo.path && <div className="truncate pl-5 font-mono text-[0.65rem] text-(--ui-text-tertiary)">{repo.path}</div>}
                <ul className="grid gap-0.5 pl-5">
                  {repo.groups.map(group => (
                    <li className="flex min-w-0 items-center gap-2 text-xs" key={group.id}>
                      <Codicon
                        className="shrink-0 text-(--ui-text-tertiary)"
                        name={group.isKanban ? 'project' : 'git-branch'}
                      />
                      <span className="truncate text-(--ui-text-secondary)">{group.label}</span>
                      <span className="shrink-0 text-[0.65rem] text-(--ui-text-tertiary)">
                        {laneLabel[laneKind(group)]}
                      </span>
                      {group.path && group.path !== repo.path && (
                        <span className="ml-auto min-w-0 truncate font-mono text-[0.65rem] text-(--ui-text-tertiary)">
                          {group.path}
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        )}
      </DetailSection>

      <DetailSection title={p.folders}>
        {folders.length === 0 ? (
          <DetailNote>{p.noPath}</DetailNote>
        ) : (
          <ul className="grid gap-0.5">
            {folders.map(folder => (
              <li className="flex min-w-0 items-center gap-2 text-xs" key={folder.path}>
                <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder" />
                <span className="min-w-0 truncate font-mono text-(--ui-text-secondary)">{folder.path}</span>
                {folder.label && <span className="shrink-0 text-(--ui-text-tertiary)">{folder.label}</span>}
                {folder.isPrimary && (
                  <Badge size="xs" variant="outline">
                    {p.primaryFolder}
                  </Badge>
                )}
              </li>
            ))}
          </ul>
        )}
      </DetailSection>
    </section>
  )
}

function DetailSection({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <section className="space-y-2">
      <h3 className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{title}</h3>
      {children}
    </section>
  )
}

function DetailNote({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-(--ui-text-tertiary)">{children}</p>
}

function SessionRow({
  onOpen,
  session,
  status
}: {
  onOpen: (sessionId: string, event: React.MouseEvent) => void
  session: SessionInfo
  status?: LiveSessionState
}) {
  const { t } = useI18n()
  const p = t.projects
  const title = session.title?.trim() || session.preview?.trim() || p.untitledSession
  const when = session.last_active || session.started_at

  return (
    <RowButton
      className="row-hover flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1 text-left text-xs text-(--ui-text-secondary) hover:text-foreground"
      onClick={event => onOpen(session.id, event)}
    >
      <span className="min-w-0 flex-1 truncate">{title}</span>
      {status && (
        <Badge size="xs" variant={STATUS_BADGE[status]}>
          {p.status[status]}
        </Badge>
      )}
      {when ? (
        <span className="shrink-0 text-[0.65rem] tabular-nums text-(--ui-text-tertiary)">
          {fmtDayTime.format(new Date(when * 1000))}
        </span>
      ) : null}
    </RowButton>
  )
}
