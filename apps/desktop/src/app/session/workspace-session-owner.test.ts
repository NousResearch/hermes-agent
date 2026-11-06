import { afterEach, describe, expect, it, vi } from 'vitest'

import { $newChatProfile, $newChatRoute, resolveNewChatOwnerRoute } from '@/store/profile'
import { $currentBranch, $newChatWorkspaceTarget, setCurrentBranch, setNewChatWorkspaceTarget } from '@/store/session'

import { startWorkspaceSession } from './workspace-session-target'

const requestGatewayForAgent = vi.hoisted(() => vi.fn())

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent
}))

describe('startWorkspaceSession for a project another gateway owns', () => {
  afterEach(() => {
    setCurrentBranch('')
    setNewChatWorkspaceTarget(undefined)
    $newChatRoute.set(null)
    $newChatProfile.set(null)
    vi.clearAllMocks()
  })

  it('creates the draft on the owner and resolves the folder there, not on the window gateway', async () => {
    requestGatewayForAgent.mockResolvedValue({ branch: 'feature', cwd: '/code/app' })
    const requestGateway = vi.fn()
    const followActiveSessionCwd = vi.fn()
    const onExplicitWorkspace = vi.fn()

    startWorkspaceSession({
      activeSessionIdRef: { current: null },
      followActiveSessionCwd,
      onExplicitWorkspace,
      owner: { connectionId: 'studio', profile: 'research' },
      path: '/code/app',
      requestGateway,
      startFreshSessionDraft: options => setNewChatWorkspaceTarget(options?.workspaceTarget)
    })

    await vi.waitFor(() => expect($currentBranch.get()).toBe('feature'))

    expect(resolveNewChatOwnerRoute()).toEqual({ connectionId: 'studio', profile: 'research' })
    expect($newChatWorkspaceTarget.get()).toBe('/code/app')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('studio', 'research', 'config.get', {
      key: 'project',
      cwd: '/code/app',
      profile: 'research'
    })
    // The window's own gateway and project tree have a different (or no) project at this path.
    expect(requestGateway).not.toHaveBeenCalled()
    expect(followActiveSessionCwd).not.toHaveBeenCalled()
    expect(onExplicitWorkspace).not.toHaveBeenCalled()
  })
})
