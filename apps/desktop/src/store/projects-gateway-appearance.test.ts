import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { $activeGatewayProfile } from '@/store/profile'

import { gatewayProjectLook } from './gateway-project-looks'
import { $projects, $projectTree, setProjectAppearance } from './projects'

vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

vi.mock('@/store/gateway', () => ({
  $gateway: atom(null),
  activeGateway: vi.fn(),
  activeGatewayConnectionId: vi.fn(() => null),
  ensureActiveGatewayOpen: vi.fn(),
  isActivePrimary: vi.fn(() => true),
  requestGatewayForAgent: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getProfiles: vi.fn(),
  hermesApi: vi.fn(),
  setApiRequestProfile: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))

const gw = await import('@/store/gateway')
const requestGatewayForAgent = vi.mocked(gw.requestGatewayForAgent)

const STUDIO = { connectionId: 'studio', profile: 'research' }

// The window's own gateway: no write for another gateway's project may land here.
const windowGateway = { connectionState: 'open', request: vi.fn(async () => ({ project: null })) }

// A project only the Studio gateway holds, as withGatewayProjectRows builds it.
const studioApp = {
  connectionId: 'studio',
  id: JSON.stringify(['gateway-project', 'studio', '/code/app']),
  isAuto: true,
  label: 'app · Studio',
  path: '/code/app',
  profile: 'research',
  repos: [],
  sessionCount: 1
} as SidebarProjectTree

const remoteProject = (fields: Record<string, unknown>) => ({
  archived: false,
  color: null,
  folders: [{ path: '/code/app' }],
  icon: null,
  id: 'p_remote',
  name: 'app',
  primary_path: '/code/app',
  ...fields
})

function serveStudio(projects: unknown[]) {
  requestGatewayForAgent.mockImplementation((async (
    _connectionId: string,
    _profile: string,
    method: string,
    params: Record<string, unknown>
  ) => {
    if (method === 'projects.list') {
      return { active_id: null, projects }
    }

    return { project: remoteProject({ ...params, id: (params.id as string) || 'p_new' }) }
  }) as never)
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(gw.activeGateway).mockReturnValue(windowGateway as never)
  $activeGatewayProfile.set('default')
  $projects.set([])
  $projectTree.set([])
})

describe("appearance of another gateway's project", () => {
  it('creates the project on the owning gateway profile, not on this window', async () => {
    serveStudio([])

    await setProjectAppearance(studioApp, { color: '#f43f5e' })

    expect(requestGatewayForAgent.mock.calls.map(call => call.slice(0, 4))).toEqual([
      ['studio', 'research', 'projects.list', { profile: 'research' }],
      [
        'studio',
        'research',
        'projects.create',
        {
          color: '#f43f5e',
          folders: ['/code/app'],
          icon: undefined,
          name: 'app',
          primary_path: '/code/app',
          profile: 'research'
        }
      ]
    ])
    expect(windowGateway.request).not.toHaveBeenCalled()
    expect($projects.get()).toEqual([])
    // The sidebar paints the owner's record back onto the project.
    expect(gatewayProjectLook(STUDIO, '/code/app')).toEqual({ color: '#f43f5e', icon: null, id: 'p_new' })
  })

  it("patches the owner's existing project for the folder instead of creating a second", async () => {
    serveStudio([remoteProject({ color: '#f43f5e' })])

    await setProjectAppearance(studioApp, { icon: 'rocket' })

    expect(requestGatewayForAgent.mock.calls.at(-1)?.slice(0, 4)).toEqual([
      'studio',
      'research',
      'projects.update',
      { icon: 'rocket', id: 'p_remote', profile: 'research' }
    ])
    expect(windowGateway.request).not.toHaveBeenCalled()
  })
})
