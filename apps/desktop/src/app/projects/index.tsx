import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { ErrorState } from '@/components/ui/error-state'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $dismissedAutoProjectIds } from '@/store/layout'
import {
  $activeProjectId,
  $projects,
  $projectsRpcAvailable,
  $projectTree,
  $projectTreeLoading,
  goToProject,
  openProjectCreate,
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

// The Projects cockpit: a first-class page over the existing project caches.
// It never moves focus on its own — mounting, refreshing, and background tree
// updates only repaint; the user's clicks are the only navigation.
export function ProjectsView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: ProjectsViewProps) {
  const { t } = useI18n()
  const p = t.projects
  const navigate = useNavigate()
  const { search } = useLocation()
  const tree = useStore($projectTree)
  const infos = useStore($projects)
  const activeProjectId = useStore($activeProjectId)
  const treeLoading = useStore($projectTreeLoading)
  const rpcAvailable = useStore($projectsRpcAvailable)
  const dotStates = useStore($sessionDotStateById)
  const removedIds = useStore($removedSessionIds)
  const dismissedAutoProjectIds = useStore($dismissedAutoProjectIds)
  const [query, setQuery] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [refreshedOnce, setRefreshedOnce] = useState(false)

  const refresh = useCallback(async () => {
    setRefreshing(true)

    try {
      // Both actions are best-effort and keep the cached atoms on failure.
      await Promise.all([refreshProjects(), refreshProjectTree()])
    } finally {
      setRefreshing(false)
      setRefreshedOnce(true)
    }
  }, [])

  useRefreshHotkey(() => void refresh())

  useEffect(() => {
    void refresh()
  }, [refresh])

  const projects = useMemo(
    () => cockpitProjects(tree, activeProjectId, dismissedAutoProjectIds),
    [activeProjectId, dismissedAutoProjectIds, tree]
  )

  const visibleProjects = useMemo(() => filterCockpitProjects(projects, query), [projects, query])
  const selectedId = new URLSearchParams(search).get(PROJECT_QUERY_PARAM)
  const selected = projects.find(project => project.id === selectedId) ?? null
  const selectedInfo = selected ? infos.find(info => info.id === selected.id) : undefined
  const { failed: sessionsFailed, hydrated } = useProjectSessions(selected)

  const { active, recent } = useMemo(
    () =>
      selected ? splitActiveSessions(projectSessionList(selected, hydrated, removedIds), dotStates) : EMPTY_SPLIT,
    [dotStates, hydrated, removedIds, selected]
  )

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
      if (!refreshedOnce || treeLoading || rpcAvailable === null) {
        return <PageLoader label={p.loading} />
      }

      return (
        <div className="grid h-full place-items-center px-6">
          <div className="flex flex-col items-center gap-3">
            <EmptyState className="min-h-0" description={p.emptyDesc} title={p.emptyTitle} />
            <Button onClick={openProjectCreate} size="sm" variant="secondary">
              {p.newProject}
            </Button>
          </div>
        </div>
      )
    }

    return (
      <MasterDetail>
        <ListColumn>
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
              sessionsFailed={sessionsFailed}
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
        <Tip label={refreshing ? p.refreshing : p.refresh}>
          <Button
            aria-label={refreshing ? p.refreshing : p.refresh}
            className="text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            disabled={refreshing}
            onClick={() => void refresh()}
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
