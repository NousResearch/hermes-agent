import { beforeEach, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'

import {
  $codingWorkspaceDrafts,
  codingWorkspaceKey,
  inspectCodingWorkspace,
  prepareCodingWorkspace,
  resetCodingWorkspaceDraft,
  setCodingWorkspaceIntent
} from './coding-workspaces'

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))
const owner = { connectionId: 'local', profile: 'coder', draftKey: 'draft-a' }
const intent = { path: '/repo', mode: 'worktree' as const }
const prepared = { cwd: '/repo/.worktrees/task', projectId: 'p', branch: 'hermes/task', repoRoot: '/repo' }
beforeEach(() => {
  vi.resetAllMocks()
  $codingWorkspaceDrafts.set({})
})

it('does no RPC before opt-in; inspection is read-only and prepare single-flight/retry-safe', async () => {
  expect(await prepareCodingWorkspace(owner)).toBeNull()
  expect(requestGatewayForAgent).not.toHaveBeenCalled()
  setCodingWorkspaceIntent(owner, intent)
  expect(requestGatewayForAgent).not.toHaveBeenCalled()
  vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({
    path: '/repo',
    repoRoot: '/repo',
    branch: 'main',
    dirty: true,
    worktrees: []
  })
  await inspectCodingWorkspace(owner)
  expect(vi.mocked(requestGatewayForAgent).mock.calls[0].slice(0, 3)).toEqual([
    'local',
    'coder',
    'projects.workspace.inspect'
  ])
  let finish!: (value: unknown) => void
  vi.mocked(requestGatewayForAgent).mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const a = prepareCodingWorkspace(owner)
  const b = prepareCodingWorkspace(owner)
  finish(prepared)
  expect(await a).toEqual(prepared)
  expect(await b).toEqual(prepared)
  expect(await prepareCodingWorkspace(owner)).toEqual(prepared)
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(2)
})

it('late inspection cannot publish over newer intent or another owner', async () => {
  setCodingWorkspaceIntent(owner, intent)
  let finish!: (value: unknown) => void
  vi.mocked(requestGatewayForAgent).mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const pending = inspectCodingWorkspace(owner)
  setCodingWorkspaceIntent(owner, { path: '/other', mode: 'current' })
  const other = { ...owner, profile: 'other' }
  setCodingWorkspaceIntent(other, intent)
  finish({ path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false, worktrees: [] })
  await expect(pending).rejects.toThrow('changed')
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].inspection).toBeUndefined()
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(other)].status).toBe('idle')
})

it('New chat rotates a reused key and cannot adopt a pending prior preparation', async () => {
  setCodingWorkspaceIntent(owner, intent)
  const firstId = $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].requestId
  let finish!: (value: unknown) => void
  vi.mocked(requestGatewayForAgent).mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const previous = prepareCodingWorkspace(owner)
  resetCodingWorkspaceDraft(owner)
  setCodingWorkspaceIntent(owner, { path: '/different', mode: 'current' })
  vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({ ...prepared, cwd: '/different' })
  expect(await prepareCodingWorkspace(owner)).toMatchObject({ cwd: '/different' })
  finish(prepared)
  await expect(previous).rejects.toThrow('changed')
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].prepared?.cwd).toBe('/different')
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].requestId).not.toBe(firstId)
})

it('read-only non-Git inspection selects folder mode without initializing Git', async () => {
  setCodingWorkspaceIntent(owner, intent)
  vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({
    path: '/repo',
    repoRoot: null,
    branch: null,
    dirty: false,
    worktrees: []
  })
  await inspectCodingWorkspace(owner)
  expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].intent?.mode).toBe('folder')
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
})
