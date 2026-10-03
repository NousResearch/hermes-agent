import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as Hermes from '@/hermes'
import type * as DesktopGit from '@/lib/desktop-git'
import type * as GatewayStore from '@/store/gateway'

// The cockpit beside the REAL sidebar and projects store: only the transport,
// config read and filesystem crawl are faked, so every consumer that reacts to
// the sidebar's grouping runs as it does in the app.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  activeGateway: vi.fn(),
  ensureActiveGatewayOpen: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof Hermes>()),
  getHermesConfig: vi.fn(async () => ({})),
  hermesApi: vi.fn()
}))

const scanRepos = vi.hoisted(() => vi.fn(async () => []))

vi.mock('@/lib/desktop-git', async importOriginal => ({
  ...(await importOriginal<typeof DesktopGit>()),
  desktopGit: () => ({ scanRepos })
}))

const gatewayStore = await import('@/store/gateway')
const hermes = await import('@/hermes')
const { SidebarProvider } = await import('@/components/ui/sidebar')
const { $sidebarAgentsGrouped, setSidebarAgentsGrouped } = await import('@/store/layout')
const { $activeGatewayProfile, setShowAllProfiles } = await import('@/store/profile')
const { $projectScope, ALL_PROJECTS } = await import('@/store/project-scope')

const {
  $projects,
  $projectsOwner,
  $projectsReadStatus,
  $projectsRpcAvailableByOwner,
  $projectTree,
  $projectTreeOwner,
  $projectTreeReadStatus
} = await import('@/store/projects')

const { $connection, $gatewayState } = await import('@/store/session')
const { ChatSidebar } = await import('@/app/chat/sidebar')
const { ProjectsView } = await import('.')

type Respond = (method: string, params: Record<string, unknown>) => Promise<unknown>

const PROJECT_READS = new Set(['projects.list', 'projects.project_sessions', 'projects.tree'])

const atlasTree = {
  active_id: null,
  errors: [],
  projects: [{ id: 'p_atlas', label: 'Atlas', path: '/work/atlas', previewSessions: [], repos: [], sessionCount: 0 }],
  scoped_session_ids: []
}

const atlasList = {
  active_id: null,
  projects: [
    {
      archived: false,
      board_slug: null,
      color: null,
      created_at: 0,
      description: null,
      folders: [],
      icon: null,
      id: 'p_atlas',
      name: 'Atlas',
      primary_path: '/work/atlas',
      slug: 'atlas'
    }
  ]
}

const respond: Respond = async method => {
  if (method === 'projects.list') {
    return atlasList
  }

  if (method === 'projects.tree') {
    return atlasTree
  }

  if (method === 'projects.project_sessions') {
    return { project: null }
  }

  if (method === 'projects.discover_repos') {
    return { discovery_policy: {}, repos: [] }
  }

  return {}
}

function openGateway() {
  // A fresh gateway each time: repo discovery has never completed on it.
  const gateway = { connectionState: 'open', request: vi.fn<Respond>(respond) }
  vi.mocked(gatewayStore.activeGateway).mockReturnValue(gateway as never)
  vi.mocked(gatewayStore.ensureActiveGatewayOpen).mockResolvedValue(gateway as never)

  return gateway
}

const noop = () => {}

const noopAsync = async () => {}

function renderCockpitWithSidebar() {
  return render(
    <MemoryRouter initialEntries={['/projects?project=p_atlas']}>
      <SidebarProvider>
        <ChatSidebar
          currentView="projects"
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={noop}
          onRetrySessions={noopAsync}
          onTriggerCronJob={noopAsync}
        />
      </SidebarProvider>
      <ProjectsView />
    </MemoryRouter>
  )
}

const settle = () => act(() => new Promise<void>(resolve => setTimeout(resolve, 0)))

const projectWrites = (gateway: ReturnType<typeof openGateway>) =>
  gateway.request.mock.calls.map(([method]) => method).filter(m => m.startsWith('projects.') && !PROJECT_READS.has(m))

const SCENARIOS = {
  'All Profiles': () => {
    $connection.set({ baseUrl: 'http://127.0.0.1:9119', connectionId: 'local', mode: 'local' } as never)
    setShowAllProfiles(true)
  },
  local: () => $connection.set({ baseUrl: 'http://127.0.0.1:9119', connectionId: 'local', mode: 'local' } as never),
  remote: () => $connection.set({ baseUrl: 'https://box.example.test', connectionId: 'box', mode: 'remote' } as never)
} as const

beforeEach(() => {
  $activeGatewayProfile.set('default')
  setShowAllProfiles(false)
  setSidebarAgentsGrouped(false)
  $gatewayState.set('open')
  vi.mocked(hermes.hermesApi).mockResolvedValue(atlasTree)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  setShowAllProfiles(false)
  setSidebarAgentsGrouped(false)
  $projectScope.set(ALL_PROJECTS)
  $projectTree.set([])
  $projects.set([])
  $projectTreeOwner.set(null)
  $projectsOwner.set(null)
  $projectTreeReadStatus.set(null)
  $projectsReadStatus.set(null)
  $projectsRpcAvailableByOwner.set({})
  $gatewayState.set('idle')
  $connection.set(null)
})

describe('Projects cockpit "Show in sidebar" beside the mounted sidebar', () => {
  it.each(Object.keys(SCENARIOS) as (keyof typeof SCENARIOS)[])(
    'groups and scopes the sidebar without any project write (%s)',
    async scenario => {
      SCENARIOS[scenario]()
      const gateway = openGateway()
      renderCockpitWithSidebar()

      const detail = await screen.findByRole('region', { name: 'Atlas' })
      await settle()
      expect($sidebarAgentsGrouped.get()).toBe(false)

      fireEvent.click(within(detail).getByRole('button', { name: 'Show in sidebar' }))
      await waitFor(() =>
        expect(gateway.request.mock.calls.length + vi.mocked(hermes.hermesApi).mock.calls.length).toBeGreaterThan(0)
      )
      await settle()
      await settle()

      expect($sidebarAgentsGrouped.get()).toBe(true)
      expect($projectScope.get()).toBe('p_atlas')
      expect(projectWrites(gateway)).toEqual([])
      expect(scanRepos).not.toHaveBeenCalled()
    }
  )

  it.each(['local', 'remote'] as const)(
    'keeps repo discovery when the user groups the sidebar themselves (%s)',
    async scenario => {
      SCENARIOS[scenario]()
      const gateway = openGateway()
      renderCockpitWithSidebar()
      await screen.findByRole('region', { name: 'Atlas' })

      act(() => setSidebarAgentsGrouped(true))

      const expected = scenario === 'local' ? 'projects.record_repos' : 'projects.discover_repos'
      await waitFor(() => expect(projectWrites(gateway)).toContain(expected))
    }
  )
})
