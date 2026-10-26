import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '../test/deferred'

import {
  $activeSessionId,
  $connection,
  $currentCwd,
  applyConfiguredDefaultProjectDir,
  ensureDefaultWorkspaceCwd,
  getConfiguredDefaultProjectDir,
  setCurrentCwd,
  syncConfiguredDefaultProjectDir,
  workspaceCwdForNewSession
} from './session'

const project = '/home/user/projects/demo'
const local = (profile = 'default') => ({ connectionId: 'local', mode: 'local', profile }) as never

const settingsResult = (dir: null | string = null, launch = true) => ({
  defaultLabel: '/home/user',
  dir,
  resolvedCwd: dir || project,
  ...(launch ? { launchWorkspace: { cwd: project, profile: 'default' } } : {})
})

function bridge(getDefaultProjectDir = vi.fn(async () => settingsResult())) {
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
    sanitizeWorkspaceCwd: vi.fn(async (cwd: string) => ({ cwd })),
    settings: { getDefaultProjectDir }
  }
}

describe('terminal launch workspace', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $connection.set(local())
    $activeSessionId.set(null)
    $currentCwd.set('')
    applyConfiguredDefaultProjectDir(null)
  })

  afterEach(async () => {
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
    // Refresh without a bridge to clear the cached launch hint too.
    await syncConfiguredDefaultProjectDir()
    applyConfiguredDefaultProjectDir(null)
    $connection.set(null)
    $activeSessionId.set(null)
    $currentCwd.set('')
    window.localStorage.clear()
  })

  it('uses the explicit local launch directory without promoting remembered workspaces or overriding settings', async () => {
    bridge()
    setCurrentCwd('/remembered/repo')
    await ensureDefaultWorkspaceCwd()

    expect($currentCwd.get()).toBe(project)
    expect(workspaceCwdForNewSession()).toBe(project)
    expect(getConfiguredDefaultProjectDir()).toBe('')
    expect(window.localStorage.getItem('hermes.desktop.workspace-cwd')).toBe('/remembered/repo')

    bridge(vi.fn(async () => settingsResult('/configured/default')))
    await ensureDefaultWorkspaceCwd()
    expect(workspaceCwdForNewSession()).toBe('/configured/default')

    // A selected workspace stays authoritative, even when settings arrive later.
    const response = deferred<ReturnType<typeof settingsResult>>()
    bridge(vi.fn(() => response.promise))
    const seed = ensureDefaultWorkspaceCwd()
    $activeSessionId.set('live')
    $currentCwd.set('/selected/worktree')
    response.resolve(settingsResult())
    await seed
    expect($currentCwd.get()).toBe('/selected/worktree')
  })

  it('keeps implicit launches detached and scopes the terminal directory to its local launch profile', async () => {
    bridge()
    await ensureDefaultWorkspaceCwd()
    expect(workspaceCwdForNewSession()).toBe(project)

    for (const connection of [
      { connectionId: 'server', mode: 'remote', profile: 'default', baseUrl: 'https://gateway.example' },
      local('other'),
      null
    ]) {
      $connection.set(connection as never)
      await ensureDefaultWorkspaceCwd()
      expect(workspaceCwdForNewSession()).toBe('')
      expect($currentCwd.get()).toBe('')
    }

    $connection.set(local())
    await ensureDefaultWorkspaceCwd()
    expect(workspaceCwdForNewSession()).toBe(project)

    // Older bridges and ordinary app launches have resolvedCwd but no explicit launch hint.
    bridge(vi.fn(async () => settingsResult(null, false)))
    setCurrentCwd('/remembered/repo')
    await ensureDefaultWorkspaceCwd()
    expect(workspaceCwdForNewSession()).toBe('')
  })
})
