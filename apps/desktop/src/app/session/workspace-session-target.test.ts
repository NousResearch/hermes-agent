import { afterEach, describe, expect, it, vi } from 'vitest'

import { $projectScope, $projectTree, ALL_PROJECTS } from '@/store/projects'
import {
  $currentBranch,
  $currentCwd,
  $newChatWorkspaceTarget,
  setCurrentBranch,
  setCurrentCwd,
  setNewChatWorkspaceTarget
} from '@/store/session'

import { deferred } from '../../test/deferred'

import { startWorkspaceSession } from './workspace-session-target'

describe('startWorkspaceSession', () => {
  afterEach(() => {
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    setCurrentBranch('')
    setCurrentCwd('')
    setNewChatWorkspaceTarget(undefined)
    vi.restoreAllMocks()
  })

  it('keeps a newer sidebar target when an older project lookup resolves', async () => {
    const first = deferred<{ branch?: string; cwd?: string }>()
    const second = deferred<{ branch?: string; cwd?: string }>()

    const requestGateway = vi
      .fn()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)

    const activeSessionIdRef = { current: null }

    const startFreshSessionDraft = vi.fn((options?: { workspaceTarget: null | string }) => {
      setNewChatWorkspaceTarget(options?.workspaceTarget)
      setCurrentCwd(options?.workspaceTarget || '')
    })

    const followActiveSessionCwd = vi.fn()

    startWorkspaceSession({
      activeSessionIdRef,
      followActiveSessionCwd,
      path: '/workspace-a',
      requestGateway,
      startFreshSessionDraft
    })
    startWorkspaceSession({
      activeSessionIdRef,
      followActiveSessionCwd,
      path: '/workspace-b',
      requestGateway,
      startFreshSessionDraft
    })

    first.resolve({ branch: 'stale', cwd: '/normalized-a' })
    await first.promise
    await Promise.resolve()

    expect($newChatWorkspaceTarget.get()).toBe('/workspace-b')
    expect($currentCwd.get()).toBe('/workspace-b')
    expect($currentBranch.get()).not.toBe('stale')

    second.resolve({ branch: 'main', cwd: '/normalized-b' })
    await second.promise
    await Promise.resolve()

    expect($newChatWorkspaceTarget.get()).toBe('/normalized-b')
    expect($currentCwd.get()).toBe('/normalized-b')
    expect($currentBranch.get()).toBe('main')
  })

  it('keeps a Home session detached instead of inheriting the active project', () => {
    $projectTree.set([
      {
        id: 'p_app',
        label: 'App',
        path: '/repo/app',
        repos: [{ groups: [], id: '/repo/app', label: 'app', path: '/repo/app', sessionCount: 0 }],
        sessionCount: 0
      }
    ])
    $projectScope.set('p_app')

    const requestGateway = vi.fn()
    const startFreshSessionDraft = vi.fn()

    startWorkspaceSession({
      activeSessionIdRef: { current: null },
      path: null,
      requestGateway,
      startFreshSessionDraft
    })

    expect(startFreshSessionDraft).toHaveBeenCalledWith({ workspaceTarget: null })
    expect(requestGateway).not.toHaveBeenCalled()
  })
})
