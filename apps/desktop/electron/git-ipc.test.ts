/**
 * `hermes:git:repoStatus` shares one in-flight probe per repo path. The composer
 * rail re-requests on every workspace-change tick and every window/focus change,
 * so the guard is what keeps a slow probe (huge repo, cold index, network cwd)
 * from stacking one git process per request.
 *
 * The large-repo fixture elsewhere covers the expensive-diff cutoff; it stays
 * green if this guard is deleted, so it cannot protect this seam.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const harness = vi.hoisted(() => ({
  handlers: new Map<string, (...args: unknown[]) => unknown>(),
  repoStatus: vi.fn()
}))

vi.mock('electron', () => ({
  ipcMain: {
    handle: (channel: string, handler: (...args: unknown[]) => unknown) => harness.handlers.set(channel, handler)
  }
}))

vi.mock('./git-repo-scan', () => ({ scanGitRepos: vi.fn(async () => []) }))

vi.mock('./git-review-ops', () => ({
  fileDiffVsHead: vi.fn(),
  repoStatus: harness.repoStatus,
  reviewCommit: vi.fn(),
  reviewCommitContext: vi.fn(),
  reviewCreatePr: vi.fn(),
  reviewDiff: vi.fn(),
  reviewList: vi.fn(),
  reviewPrList: vi.fn(),
  reviewPush: vi.fn(),
  reviewRevert: vi.fn(),
  reviewRevParse: vi.fn(),
  reviewShipInfo: vi.fn(),
  reviewStage: vi.fn(),
  reviewUnstage: vi.fn()
}))

vi.mock('./git-worktree-ops', () => ({
  addWorktree: vi.fn(),
  listBaseBranches: vi.fn(),
  listBranches: vi.fn(),
  listWorktrees: vi.fn(),
  removeWorktree: vi.fn(),
  switchBranch: vi.fn()
}))

import { registerGitIpc } from './git-ipc'

registerGitIpc({ resolveGitBinary: () => 'git', resolveGhBinary: () => 'gh' })

const repoStatus = (repoPath?: unknown) =>
  harness.handlers.get('hermes:git:repoStatus')!({}, repoPath) as Promise<unknown>

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, resolve, reject }
}

beforeEach(() => {
  harness.repoStatus.mockReset()
})

describe('hermes:git:repoStatus single-flight', () => {
  // Two overlapping calls for the same path must share one underlying probe:
  // the second caller used to spawn its own git process.
  it('shares one probe between overlapping calls for the same path', async () => {
    const probe = deferred<string>()

    harness.repoStatus.mockReturnValueOnce(probe.promise)

    const first = repoStatus('/repo/a')
    const second = repoStatus('/repo/a')

    expect(harness.repoStatus).toHaveBeenCalledTimes(1)
    expect(harness.repoStatus).toHaveBeenCalledWith('/repo/a', 'git')

    probe.resolve('status-a')

    await expect(first).resolves.toBe('status-a')
    await expect(second).resolves.toBe('status-a')
  })

  // Sharing must end when the probe settles: a request that arrives afterwards
  // is a fresh read, not a replay of the previous result.
  it('probes again after the shared probe settles', async () => {
    harness.repoStatus.mockResolvedValueOnce('first').mockResolvedValueOnce('second')

    await expect(repoStatus('/repo/a')).resolves.toBe('first')
    await expect(repoStatus('/repo/a')).resolves.toBe('second')
    expect(harness.repoStatus).toHaveBeenCalledTimes(2)
  })

  // The map is keyed by path, so a second repo in the same window must not be
  // blocked behind the first one's slow probe.
  it('keeps different paths on separate probes', async () => {
    const probeA = deferred<string>()
    const probeB = deferred<string>()

    harness.repoStatus.mockReturnValueOnce(probeA.promise).mockReturnValueOnce(probeB.promise)

    const first = repoStatus('/repo/a')
    const second = repoStatus('/repo/b')

    expect(harness.repoStatus).toHaveBeenCalledTimes(2)

    probeB.resolve('status-b')
    probeA.resolve('status-a')

    await expect(first).resolves.toBe('status-a')
    await expect(second).resolves.toBe('status-b')
  })

  // A rejected probe is shared like a resolved one, and must also be dropped:
  // a transient git failure cannot leave the path permanently stuck on a dead
  // promise.
  it('shares a rejection, then probes again on the next call', async () => {
    const probe = deferred<string>()

    harness.repoStatus.mockReturnValueOnce(probe.promise).mockResolvedValueOnce('recovered')

    const first = repoStatus('/repo/a')
    const second = repoStatus('/repo/a')

    expect(harness.repoStatus).toHaveBeenCalledTimes(1)

    const firstCheck = expect(first).rejects.toThrow('git exploded')
    const secondCheck = expect(second).rejects.toThrow('git exploded')

    probe.reject(new Error('git exploded'))

    await firstCheck
    await secondCheck
    await expect(repoStatus('/repo/a')).resolves.toBe('recovered')
    expect(harness.repoStatus).toHaveBeenCalledTimes(2)
  })

  // A non-string path (the renderer may send nothing while the workspace is
  // still resolving) keys as the empty string, so those calls coalesce too.
  it('coalesces calls that carry no repo path', async () => {
    const probe = deferred<null>()

    harness.repoStatus.mockReturnValueOnce(probe.promise)

    const first = repoStatus(undefined)
    const second = repoStatus(undefined)

    expect(harness.repoStatus).toHaveBeenCalledTimes(1)
    probe.resolve(null)
    await expect(first).resolves.toBeNull()
    await expect(second).resolves.toBeNull()
  })
})
