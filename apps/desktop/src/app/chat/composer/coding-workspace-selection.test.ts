import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $codingWorkspaceDrafts, codingWorkspaceKey } from '@/store/coding-workspaces'

import { selectCodingWorkspaceIntent } from './coding-workspace-selection'

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: (...a: unknown[]) => mocks.request(...a) }))
const owner = { connectionId: 'local', profile: 'coder', draftKey: 'draft:folder' }
beforeEach(() => { $codingWorkspaceDrafts.set({}); vi.clearAllMocks() })
describe('workspace selection', () => {
  it('uses Project folder for non-Git paths and only inspects while selecting', async () => {
    mocks.request.mockResolvedValue({ path: '/folder', repoRoot: null, branch: null, dirty: false, worktrees: [] })
    await selectCodingWorkspaceIntent(owner, { path: '/folder', mode: 'worktree' })
    const draft = $codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)]
    expect(draft.intent?.mode).toBe('folder')
    expect(draft.inspection?.repoRoot).toBeNull()
    expect(draft.prepared).toBeUndefined()
    expect(mocks.request.mock.calls.every(call => call[2] === 'projects.workspace.inspect')).toBe(true)
  })
})
