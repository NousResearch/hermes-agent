import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { ErrorBanner, ErrorState } from '@/components/ui/error-state'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $activeConnectionId } from '@/store/connections'
import { $dismissedAutoProjectIds } from '@/store/layout'
import { $profileScope, ALL_PROFILES } from '@/store/profile'
import {
  $activeProjectId,
  $projects,
  $projectsRpcAvailable,
  $projectTree,
  goToProject,
  refreshProjects,
  refreshProjectTree
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

/** The latest settled list/tree read, for the owner (connection + profile
 *  view) it was made under — another owner's failure is not this one's. */
interface ProjectsLoad {
  failed: boolean
  owner: string
}

// The Projects cockpit: a first-class page over the existing project caches.
// It never moves focus on its own — mounting, refreshing, and background tree
// updates only repaint; the user's clicks are the only navigation.
export function ProjectsView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: ProjectsViewProps) {
  const { t } = useI18n()
  const p = t.projects
  const connectionId = useStore($activeConnectionId)
  const profileScope = useStore($profileScope)
  const allProfiles = profileScope === ALL_PROFILES
  const owner = `${connectionId ?? ''}\u0000${profileScope}`
  const navigate = useNavigate()
  const { search } = useLocation()
  const tree = useStore($projectTree)
  const infos = useStore($projects)
  const activeProjectId = useStore($activeProjectId)
  const rpcAvailable = useStore($projectsRpcAvailable)
  const dotStates = useStore($sessionDotStateById)
  const removedIds = useStore($removedSessionIds)
  const dismissedAutoProjectIds = useStore($dismissedAutoProjectIds)
  const [query, setQuery] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [load, setLoad] = useState<null | ProjectsLoad>(null)
  const [sessionsRefreshToken, setSessionsRefreshToken] = useState(0)
  const refreshRun = useRef(0)

  // Re-read on mount and whenever the owner changes. Both reads keep the
  // cached atoms on failure and settle (never reject), so every run ends in a
  // ready, empty, or error state — never an open-ended loader. Only the newest
  // run may publish its outcome.
  const refresh = useCallback(async () => {
    const run = ++refreshRun.current
    setRefreshing(true)

    // `projects.list` answers for one profile; All Profiles has only the tree.
    const [listOk, treeOk] = await Promise.all([allProfiles || refreshProjects(), refreshProjectTree()])

    if (run === refreshRun.current) {
      setRefreshing(false)
      setLoad({ failed: !listOk || !treeOk, owner })
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

  const loaded = load?.owner === owner ? load : null

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

      // The search header (and its refresh) is hidden with nothing to search,
      // so the way out lives in the body.
      const retry = (
        <Button disabled={refreshing} onClick={reload} size="sm" variant="secondary">
          {refreshing ? <TitlebarIcon name="loading" spinning /> : <TitlebarIcon name="refresh" />}
          {loaded.failed ? t.common.retry : refreshLabel}
        </Button>
      )

      if (loaded.failed) {
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
          {loaded?.failed && <ErrorBanner className="mb-2">{p.partialFailed}</ErrorBanner>}
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
              onShowInSidebar={() => goToProject(selected.id)}
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
