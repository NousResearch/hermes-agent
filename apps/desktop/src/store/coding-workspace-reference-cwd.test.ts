import { beforeEach, expect, test, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'

import {
  $codingWorkspaceDrafts, captureCodingWorkspaceReferenceCwd, codingWorkspaceKey,
  resetCodingWorkspaceDraft, setCodingWorkspaceIntent
} from './coding-workspaces'

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))
const owner = { connectionId: 'fixture-connection', profile: 'coder', draftKey: 'fixture-draft' }
beforeEach(() => { vi.resetAllMocks(); $codingWorkspaceDrafts.set({}) })

test('first-send and slash precreation retain the original owner base across failed-send retries', async () => {
  setCodingWorkspaceIntent(owner, { path: '/selected', mode: 'worktree' })
  vi.mocked(requestGatewayForAgent).mockResolvedValue({ items: [], sourceCwd: '/original' })
  expect(await captureCodingWorkspaceReferenceCwd(owner)).toBe(true)
  expect(requestGatewayForAgent).toHaveBeenCalledWith(owner.connectionId, owner.profile, 'complete.path', { profile: 'coder', word: '' })
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].referenceCwd).toBe('/original')
  expect(await captureCodingWorkspaceReferenceCwd(owner, '/prepared')).toBe(true)
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].referenceCwd).toBe('/original')
})

test('an inherited composer folder is resolved by the same owning profile as completion', async () => {
  setCodingWorkspaceIntent(owner, { path: '/selected', mode: 'current' })
  vi.mocked(requestGatewayForAgent).mockResolvedValue({ items: [], sourceCwd: '/profile-base' })
  expect(await captureCodingWorkspaceReferenceCwd(owner, '/inherited', () => true, false)).toBe(true)
  expect(requestGatewayForAgent).toHaveBeenCalledWith(owner.connectionId, owner.profile, 'complete.path', {
    profile: 'coder', word: '', cwd: '/inherited', cwd_explicit: false
  })
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].referenceCwd).toBe('/profile-base')
})

test('scoped explicit bases need no inspection; stale inspection cannot rewrite a fresh draft', async () => {
  setCodingWorkspaceIntent(owner, { path: '/selected', mode: 'current' })
  expect(await captureCodingWorkspaceReferenceCwd(owner, '/scoped')).toBe(true)
  expect(requestGatewayForAgent).not.toHaveBeenCalled()
  resetCodingWorkspaceDraft(owner)
  setCodingWorkspaceIntent(owner, { path: '/next', mode: 'current' })
  let finish!: (value: unknown) => void
  vi.mocked(requestGatewayForAgent).mockImplementation(() => new Promise(resolve => { finish = resolve }))
  const pending = captureCodingWorkspaceReferenceCwd(owner)
  resetCodingWorkspaceDraft(owner)
  finish({ items: [], sourceCwd: '/retired' })
  expect(await pending).toBe(false)
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].referenceCwd).toBeUndefined()
})
