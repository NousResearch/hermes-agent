import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { ErrorBanner, ErrorState } from '@/components/ui/error-state'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import type { ProjectInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $dismissedAutoProjectIds } from '@/store/layout'
import { $profileScope, ALL_PROFILES } from '@/store/profile'
import {
  $activeProjectId,
  $projects,
  $projectsOwner,
  $projectsOwnerKey,
  $projectsReadStatus,
  $projectsRpcAvailableByOwner,
  $projectTree,
  $projectTreeOwner,
  $projectTreeReadStatus,
  type ProjectsReadOutcome,
  type ProjectsReadStatus,
  refreshProjects,
  refreshProjectTree,
  showProjectInSidebar
} from '@/store/projects'
import { $sessionDotStateById } from '@/store/session-dot-state'
import { $removedSessionIds } from '@/store/session-removal'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { DetailColumn, ListColumn, MasterDetail } from '../master-detail'
import { openSessionFromPicker, openSessionIntentFromModifiers } from '../open-session'
import { PageSearchShell } from '../page-search-shell'
import { ARTIFACTS_ROUTE, navigateToWorkspacePage } from '../routes'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import {
  cockpitProjects,
  filterCockpitProjects,
  PROJECT_QUERY_PARAM,
  projectOverviewRoute,
  projectSessionList,
  splitActiveSessions
} from './model'
import { ProjectDetail } from './project-detail'
import { useProjectSessions } from './use-project-sessions'

interface ProjectsViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

// The Projects cockpit: a first-class page over the existing project caches.
// It never moves focus on its own — mounting, refreshing, and background tree
// updates only repaint; the user's clicks are the only navigation.
export function ProjectsView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: ProjectsViewProps) {
  const { t } = useI18n()
  const p = t.projects
  const allProfiles = useStore($profileScope) === ALL_PROFILES
  const owner = useStore($projectsOwnerKey)
  const navigate = useNavigate()
  const { search } = useLocation()
  // The caches are shared with the sidebar and keep a departed owner's rows
  // until the new owner's read lands (or forever, if it fails). Project ids
  // repeat across owners, so only rows read for THIS owner may paint here.
  const sharedTree = useStore($projectTree)
  const sharedInfos = useStore($projects)
  const tree = useStore($projectTreeOwner) === owner ? sharedTree : NO_TREE
  const infos = useStore($projectsOwner) === owner ? sharedInfos : NO_INFOS
  const activeProjectId = useStore($activeProjectId)
  // Only this owner's own evidence: another backend's missing methods say nothing here.
  const rpcAvailable = useStore($projectsRpcAvailableByOwner)[owner] ?? null
  const dotStates = useStore($sessionDotStateById)
  const removedIds = useStore($removedSessionIds)
  const dismissedAutoProjectIds = useStore($dismissedAutoProjectIds)
  const [query, setQuery] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  // The owner whose first cockpit read has settled. Until then the page is
  // loading; after it, the store's published verdicts (whoever triggered the
  // read — this page, the sidebar, a background sync) say failed/incomplete.
  const [settledOwner, setSettledOwner] = useState<null | string>(null)
  const treeStatus = useStore($projectTreeReadStatus)
  const listStatus = useStore($projectsReadStatus)
  const [sessionsRefreshToken, setSessionsRefreshToken] = useState(0)
  const refreshRun = useRef(0)

  // Re-read on mount and whenever the owner changes. Both reads keep the
  // cached atoms on failure and settle (never reject), so every run ends in a
  // ready, empty, or error state — never an open-ended loader. A read another
  // same-owner refresh took over settles with THAT read. Only the newest run
  // may settle the page, and a departed owner's run leaves it to the next.
  const refresh = useCallback(async () => {
    const run = ++refreshRun.current
    setRefreshing(true)

    // `projects.list` answers for one profile; All Profiles has only the tree.
    const [list, tree] = await Promise.all([allProfiles ? LIST_NOT_READ : refreshProjects(), refreshProjectTree()])

    if (run !== refreshRun.current) {
      return
    }

    setRefreshing(false)

    if (list !== 'departed' && tree !== 'departed') {
      setSettledOwner(owner)
    }
  }, [allProfiles, owner])

  // An explicit refresh also re-reads the selected project's sessions, even
  // when the tree reports nothing new (a retry, or a title-only change).
  const reload = () => {
    setSessionsRefreshToken(token => token + 1)
    void refresh()
  }

  useRefreshHotkey(reload)

  useEffect(() => {
    void refresh()
  }, [refresh])

  // A verdict for another owner says nothing about this one; no verdict at all
  // after a settled run means nothing landed for this owner.
  const verdict = (status: null | ProjectsReadStatus) => (status?.owner === owner ? status.outcome : 'failed')
  const treeVerdict = verdict(treeStatus)

  const loaded =
    settledOwner === owner
      ? {
          failed: treeVerdict === 'failed' || (!allProfiles && verdict(listStatus) === 'failed'),
          incomplete: treeVerdict === 'incomplete'
        }
      : null

  const projects = useMemo(
    () => cockpitProjects(tree, activeProjectId, dismissedAutoProjectIds),
    [activeProjectId, dismissedAutoProjectIds, tree]
  )

  const visibleProjects = useMemo(() => filterCockpitProjects(projects, query), [projects, query])
  const selectedId = new URLSearchParams(search).get(PROJECT_QUERY_PARAM)
  const selected = projects.find(project => project.id === selectedId) ?? null
  // `projects.list` rows belong to the live profile, not to All Profiles' merged tree.
  const selectedInfo = selected && !allProfiles ? infos.find(info => info.id === selected.id) : undefined
  const { hydrated, status: sessionsStatus } = useProjectSessions(selected, sessionsRefreshToken)

  // All Profiles can only offer the tree's short, cross-owner preview; listing
  // it as the project's sessions would be incomplete, so nothing is listed.
  const { active, recent } = useMemo(
    () =>
      selected && sessionsStatus !== 'limited'
        ? splitActiveSessions(projectSessionList(selected, hydrated, removedIds), dotStates)
        : EMPTY_SPLIT,
    [dotStates, hydrated, removedIds, selected, sessionsStatus]
  )

  const refreshLabel = refreshing ? p.refreshing : p.refresh

  const selectProject = (id: string) => navigate(projectOverviewRoute(id), { replace: true })

  const body = () => {
    if (rpcAvailable === false) {
      return (
        <div className="grid h-full place-items-center px-6">
          <ErrorState description={p.unavailableDesc} title={p.unavailableTitle} />
        </div>
      )
    }

    if (projects.length === 0) {
      if (!loaded) {
        return <PageLoader label={p.loading} />
      }

      // Nothing to show after a failed or partial read is not "no projects".
      const unsettled = loaded.failed || loaded.incomplete

      // The search header (and its refresh) is hidden with nothing to search,
      // so the way out lives in the body.
      const retry = (
        <Button disabled={refreshing} onClick={reload} size="sm" variant="secondary">
          {refreshing ? <TitlebarIcon name="loading" spinning /> : <TitlebarIcon name="refresh" />}
          {unsettled ? t.common.retry : refreshLabel}
        </Button>
      )

      if (unsettled) {
        return (
          <div className="grid h-full place-items-center px-6">
            <ErrorState description={p.loadFailedDesc} title={p.loadFailedTitle}>
              <div className="flex justify-center">{retry}</div>
            </ErrorState>
          </div>
        )
      }

      return (
        <div className="grid h-full place-items-center px-6">
          <div className="flex flex-col items-center gap-3">
            <EmptyState className="min-h-0" description={p.emptyDesc} title={p.emptyTitle} />
            {retry}
          </div>
        </div>
      )
    }

    return (
      <MasterDetail>
        <ListColumn>
          {loaded?.failed ? (
            <ErrorBanner className="mb-2">{p.partialFailed}</ErrorBanner>
          ) : (
            loaded?.incomplete && <ErrorBanner className="mb-2">{p.incompleteProfiles}</ErrorBanner>
          )}
          {visibleProjects.length === 0 ? (
            <EmptyState title={p.noMatchesTitle} />
          ) : (
            <ul className="space-y-px">
              {visibleProjects.map(project => (
                <li key={project.id}>
                  <RowButton
                    aria-current={project.id === selected?.id ? 'true' : undefined}
                    className={cn(
                      'row-hover flex w-full min-w-0 flex-col items-start rounded-md px-2 py-1.5 text-left hover:text-foreground',
                      project.id === selected?.id
                        ? 'bg-(--ui-row-active-background) text-foreground'
                        : 'text-(--ui-text-secondary)'
                    )}
                    onClick={() => selectProject(project.id)}
                  >
                    <span className="w-full truncate text-[0.78rem] font-medium">{project.label}</span>
                    <span className="w-full truncate text-[0.62rem] text-(--ui-text-tertiary)">
                      {p.sessionCount(project.sessionCount)}
                    </span>
                  </RowButton>
                </li>
              ))}
            </ul>
          )}
        </ListColumn>
        <DetailColumn>
          {selected ? (
            <ProjectDetail
              active={active}
              info={selectedInfo}
              onOpenArtifacts={() => navigateToWorkspacePage(navigate, ARTIFACTS_ROUTE)}
              onOpenSession={(sessionId, event) =>
                openSessionFromPicker(sessionId, navigate, openSessionIntentFromModifiers(event))
              }
              onShowInSidebar={() => showProjectInSidebar(selected.id)}
              project={selected}
              recent={recent}
              sessionsStatus={sessionsStatus}
            />
          ) : (
            <EmptyState description={p.selectDesc} title={p.selectTitle} />
          )}
        </DetailColumn>
      </MasterDetail>
    )
  }

  return (
    <PageSearchShell
      {...props}
      onSearchChange={setQuery}
      searchHidden={projects.length === 0}
      searchPlaceholder={p.search}
      searchTrailingAction={
        <Tip label={refreshLabel}>
          <Button
            aria-label={refreshLabel}
            className="text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            disabled={refreshing}
            onClick={reload}
            size="icon-titlebar"
            variant="ghost"
          >
            {refreshing ? <TitlebarIcon name="loading" spinning /> : <TitlebarIcon name="refresh" />}
          </Button>
        </Tip>
      }
      searchValue={query}
    >
      {body()}
    </PageSearchShell>
  )
}

const EMPTY_SPLIT = { active: [], recent: [] }
const LIST_NOT_READ: ProjectsReadOutcome = 'complete'
const NO_TREE: SidebarProjectTree[] = []
const NO_INFOS: ProjectInfo[] = []
