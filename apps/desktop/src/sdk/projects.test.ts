import { beforeEach, describe, expect, it } from 'vitest'

import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { host } from '@/sdk'
import { $activeGatewayProfile, setShowAllProfiles } from '@/store/profile'
import { $projectTree, $startWorkSessionRequest } from '@/store/projects'

import { projectsHost } from './projects'

const treeNode = (overrides: Record<string, unknown>) => ({
  id: 'p_x',
  label: 'X',
  path: '/repos/x',
  repos: [],
  sessionCount: 0,
  ...overrides,
})

beforeEach(() => {
  $activeGatewayProfile.set('default')
  setShowAllProfiles(false)
  $startWorkSessionRequest.set(null)
  $projectTree.set([
    treeNode({ id: 'p_web', label: 'Website', path: '/repos/website', sessionCount: 2 }),
    treeNode({ id: 'p_api', label: 'API', path: '/repos/api', sessionCount: 0 }),
    treeNode({ id: NO_PROJECT_ID, label: 'Home', path: null, isNoProject: true, sessionCount: 1 }),
  ] as never)
})

describe('projectsHost.list', () => {
  it('exposes active-profile projects for a picker, skipping the path-less Home bucket', () => {
    expect(projectsHost.list()).toEqual([
      { id: 'p_web', label: 'Website', path: '/repos/website', sessionCount: 2 },
      { id: 'p_api', label: 'API', path: '/repos/api', sessionCount: 0 },
    ])
  })

  it('refuses to list while viewing all profiles', () => {
    setShowAllProfiles(true)

    expect(() => projectsHost.list()).toThrow('Projects are unavailable while viewing all profiles')
  })
})

describe('projectsHost.$list', () => {
  it('mirrors list() while a single profile is active', () => {
    expect(projectsHost.$list.get()).toEqual(projectsHost.list())
  })

  it('starts empty and follows the tree as it arrives after mount', () => {
    $projectTree.set([])

    expect(projectsHost.$list.get()).toEqual([])

    $projectTree.set([treeNode({ id: 'p_web', label: 'Website', path: '/repos/website', sessionCount: 2 })] as never)

    expect(projectsHost.$list.get()).toEqual([
      { id: 'p_web', label: 'Website', path: '/repos/website', sessionCount: 2 },
    ])
  })

  it('stays empty while viewing all profiles, even with a seeded tree', () => {
    setShowAllProfiles(true)

    expect(projectsHost.$list.get()).toEqual([])
  })
})

describe('host.projects wiring', () => {
  it('exposes the project verbs on the plugin host', () => {
    expect(host.projects).toBe(projectsHost)
    expect(typeof host.projects.list).toBe('function')
    expect(typeof host.projects.openNewSession).toBe('function')
    expect(host.projects.$list).toBe(projectsHost.$list)
  })
})

describe('projectsHost.openNewSession', () => {
  it('starts a fresh draft anchored at the selected project root', () => {
    projectsHost.openNewSession({ projectId: 'p_web' })

    expect($startWorkSessionRequest.get()?.path).toBe('/repos/website')
  })

  it('accepts an explicit folder path without entering sidebar scope', () => {
    projectsHost.openNewSession({ path: ' /repos/api ' })

    expect($startWorkSessionRequest.get()?.path).toBe('/repos/api')
  })

  it('rejects an unknown project without starting anything', () => {
    expect(() => projectsHost.openNewSession({ projectId: 'missing' })).toThrow('Unknown project')

    expect($startWorkSessionRequest.get()).toBeNull()
  })

  it('needs a projectId or a folder path', () => {
    expect(() => projectsHost.openNewSession({})).toThrow('Pass a projectId or a folder path')

    expect($startWorkSessionRequest.get()).toBeNull()
  })

  it('rejects a project with no folder without starting anything', () => {
    $projectTree.set([treeNode({ id: 'p_empty', label: 'Empty', path: '  ' })] as never)

    expect(() => projectsHost.openNewSession({ projectId: 'p_empty' })).toThrow('Project has no folder')

    expect($startWorkSessionRequest.get()).toBeNull()
  })

  it('refuses to start while viewing all profiles', () => {
    setShowAllProfiles(true)

    expect(() => projectsHost.openNewSession({ projectId: 'p_web' })).toThrow(
      'Projects are unavailable while viewing all profiles'
    )

    expect($startWorkSessionRequest.get()).toBeNull()
  })
})
