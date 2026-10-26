import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import {
  $connection,
  applyConfiguredDefaultProjectDir,
  setCurrentCwdExplicit,
  syncConfiguredDefaultProjectDir
} from '@/store/session'

vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

import { desktopSessionCreateParams } from './session-create-request'

const project = '/home/user/projects/demo'

describe('launch cwd at session creation', () => {
  beforeEach(async () => {
    $projectScope.set(ALL_PROJECTS)
    $connection.set({ connectionId: 'local', mode: 'local', profile: 'default' } as never)
    setCurrentCwdExplicit(false)
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      settings: {
        getDefaultProjectDir: async () => ({
          dir: null,
          resolvedCwd: project,
          launchWorkspace: { cwd: project, profile: 'default' }
        })
      }
    }
    await syncConfiguredDefaultProjectDir()
  })

  afterEach(async () => {
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
    await syncConfiguredDefaultProjectDir()
    $connection.set(null)
    setCurrentCwdExplicit(false)
    $projectScope.set(ALL_PROJECTS)
  })

  it('ships the launch cwd only to its local owner, including when a draft targets another connection or profile', async () => {
    const local = await desktopSessionCreateParams(project, { connectionId: 'local', profile: 'default' })
    expect(local).toMatchObject({ cwd: project, cwd_explicit: false, profile: 'default' })

    for (const route of [
      { connectionId: 'cloud', profile: 'default' },
      { connectionId: 'local', profile: 'other' },
      { connectionId: 'local', profile: 'default', targetProfile: 'other' }
    ]) {
      expect(await desktopSessionCreateParams(project, route)).not.toHaveProperty('cwd')
    }

    expect(await desktopSessionCreateParams(project, null, 'other')).not.toHaveProperty('cwd')
    $connection.set({ mode: 'remote', profile: 'default' } as never)
    expect(await desktopSessionCreateParams(project, null, 'default')).not.toHaveProperty('cwd')
  })

  it('preserves explicit selections and configured defaults, even when they equal the launch directory', async () => {
    const remote = { connectionId: 'cloud', profile: 'other' }
    setCurrentCwdExplicit(true)
    expect(await desktopSessionCreateParams(project, remote)).toMatchObject({ cwd: project, cwd_explicit: true })
    setCurrentCwdExplicit(false)
    applyConfiguredDefaultProjectDir(project)
    expect(await desktopSessionCreateParams(project, remote)).toMatchObject({ cwd: project, cwd_explicit: false })
    applyConfiguredDefaultProjectDir(null)
    expect(await desktopSessionCreateParams('/explicit/worktree', remote)).toHaveProperty('cwd', '/explicit/worktree')
    expect(await desktopSessionCreateParams('', remote)).not.toHaveProperty('cwd')
    $projectScope.set('selected-project')
    expect(await desktopSessionCreateParams(project, remote)).toHaveProperty('cwd', project)
  })
})
