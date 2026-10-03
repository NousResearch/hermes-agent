import { beforeEach, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'

import {
  $codingWorkspaceDrafts,
  codingWorkspaceKey,
  initializeCodingWorkspace,
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

it('Initialize is an explicit write that continues the same draft on the current checkout', async () => {
  setCodingWorkspaceIntent(owner, { path: '/folder', mode: 'worktree' })
  vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({ path: '/folder', repoRoot: null, branch: null, dirty: false, worktrees: [] })
  await inspectCodingWorkspace(owner)
  const before = $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]
  expect(before.intent?.mode).toBe('folder')
  vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({
    path: '/folder', repoRoot: '/folder', branch: 'main', dirty: true, branches: ['main'],
    worktrees: [{ path: '/folder', branch: 'main', isMain: true, dirty: true }]
  })
  await initializeCodingWorkspace(owner)
  expect(vi.mocked(requestGatewayForAgent).mock.calls[1].slice(0, 4)).toEqual([
    'local', 'coder', 'projects.workspace.initialize', { path: '/folder', profile: 'coder' }
  ])
  const after = $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]
  // Same draft (same request identity), now a Git project on the checkout that holds the files.
  expect(after.requestId).toBe(before.requestId)
  expect(after.status).toBe('ready')
  expect(after.intent).toEqual({ path: '/folder', mode: 'current', existingPath: undefined })
  expect(after.inspection?.repoRoot).toBe('/folder')
  expect(after.prepared).toBeUndefined()
})

it('a failed Initialize keeps the folder draft and paints the error', async () => {
  setCodingWorkspaceIntent(owner, { path: '/folder', mode: 'folder' })
  vi.mocked(requestGatewayForAgent).mockRejectedValueOnce(new Error('Folder is already inside a Git repository'))
  await expect(initializeCodingWorkspace(owner)).rejects.toThrow('already inside')
  const draft = $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]
  expect(draft.status).toBe('error')
  expect(draft.error).toContain('already inside')
  expect(draft.intent?.mode).toBe('folder')
})
