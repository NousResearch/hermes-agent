import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $gitAvailable,
  $gitBranches,
  $gitError,
  $gitLog,
  $gitStashes,
  $gitStatus,
  applyHunk,
  checkoutBranch,
  commit,
  createBranch,
  refreshBranches,
  refreshGitStatus,
  refreshLog,
  refreshStashes,
  stageAll,
  stageFiles,
  stashAction,
  unstageFiles
} from './git'
import { $currentCwd } from './session'

// A controllable fake of window.hermesDesktop.git.
function installGitApi(overrides: Partial<Record<string, ReturnType<typeof vi.fn>>> = {}) {
  const api = {
    status: vi.fn(),
    diff: vi.fn(),
    stage: vi.fn().mockResolvedValue({ ok: true }),
    unstage: vi.fn().mockResolvedValue({ ok: true }),
    discard: vi.fn().mockResolvedValue({ ok: true }),
    commit: vi.fn().mockResolvedValue({ ok: true }),
    push: vi.fn().mockResolvedValue({ ok: true }),
    pull: vi.fn().mockResolvedValue({ ok: true }),
    fetch: vi.fn().mockResolvedValue({ ok: true }),
    branches: vi.fn().mockResolvedValue({ ok: true, current: 'main', local: [], remote: [] }),
    checkout: vi.fn().mockResolvedValue({ ok: true }),
    createBranch: vi.fn().mockResolvedValue({ ok: true }),
    deleteBranch: vi.fn().mockResolvedValue({ ok: true }),
    log: vi.fn().mockResolvedValue({ ok: true, commits: [] }),
    commitDiff: vi.fn().mockResolvedValue({ ok: true, diff: '' }),
    stashList: vi.fn().mockResolvedValue({ ok: true, stashes: [] }),
    stashPush: vi.fn().mockResolvedValue({ ok: true }),
    stashAction: vi.fn().mockResolvedValue({ ok: true }),
    applyHunk: vi.fn().mockResolvedValue({ ok: true }),
    ...overrides
  }

  ;(globalThis as unknown as { window: { hermesDesktop: { git: typeof api } } }).window = {
    hermesDesktop: { git: api }
  } as never

  return api
}

const REPO = '/tmp/myrepo'

function statusPayload(extra: Record<string, unknown> = {}) {
  return {
    ok: true,
    root: REPO,
    branch: 'main',
    upstream: 'origin/main',
    ahead: 0,
    behind: 0,
    staged: [],
    unstaged: [],
    untracked: [],
    conflicted: [],
    ...extra
  }
}

beforeEach(() => {
  $currentCwd.set(REPO)
})

afterEach(() => {
  vi.restoreAllMocks()
  $gitStatus.set({
    root: null,
    branch: null,
    upstream: null,
    ahead: 0,
    behind: 0,
    staged: [],
    unstaged: [],
    untracked: [],
    conflicted: []
  })
  $gitAvailable.set(false)
  $gitError.set(null)
})

describe('refreshGitStatus', () => {
  it('populates the snapshot from a successful status', async () => {
    installGitApi({
      status: vi.fn().mockResolvedValue(
        statusPayload({
          ahead: 2,
          behind: 1,
          staged: [{ path: 'a.txt', status: 'modified', staged: true }],
          unstaged: [{ path: 'b.txt', status: 'modified', staged: false }],
          untracked: [{ path: 'c.txt', status: 'untracked', staged: false }]
        })
      )
    })

    await refreshGitStatus()

    const s = $gitStatus.get()
    expect($gitAvailable.get()).toBe(true)
    expect(s.branch).toBe('main')
    expect(s.ahead).toBe(2)
    expect(s.behind).toBe(1)
    expect(s.staged.map(f => f.path)).toEqual(['a.txt'])
    expect(s.unstaged.map(f => f.path)).toEqual(['b.txt'])
    expect(s.untracked.map(f => f.path)).toEqual(['c.txt'])
  })

  it('treats not-a-repo as unavailable without an error', async () => {
    installGitApi({ status: vi.fn().mockResolvedValue({ ok: false, error: 'not-a-repo' }) })

    await refreshGitStatus()

    expect($gitAvailable.get()).toBe(false)
    expect($gitError.get()).toBeNull()
  })

  it('surfaces a real git error', async () => {
    installGitApi({ status: vi.fn().mockResolvedValue({ ok: false, error: 'fatal: bad object' }) })

    await refreshGitStatus()

    expect($gitAvailable.get()).toBe(false)
    expect($gitError.get()).toBe('fatal: bad object')
  })
})

describe('mutations refresh afterward', () => {
  it('stageFiles calls stage then re-reads status', async () => {
    const api = installGitApi({
      status: vi
        .fn()
        .mockResolvedValue(statusPayload({ staged: [{ path: 'a.txt', status: 'modified', staged: true }] }))
    })

    const ok = await stageFiles(['a.txt'])

    expect(ok).toBe(true)
    expect(api.stage).toHaveBeenCalledWith(REPO, ['a.txt'])
    // status re-read after the mutation
    expect(api.status).toHaveBeenCalled()
    expect($gitStatus.get().staged.map(f => f.path)).toEqual(['a.txt'])
  })

  it('stageAll stages every unstaged + untracked + conflicted path', async () => {
    const api = installGitApi({ status: vi.fn().mockResolvedValue(statusPayload()) })
    $gitStatus.set({
      root: REPO,
      branch: 'main',
      upstream: null,
      ahead: 0,
      behind: 0,
      staged: [],
      unstaged: [{ path: 'a.txt', status: 'modified', staged: false }],
      untracked: [{ path: 'b.txt', status: 'untracked', staged: false }],
      conflicted: [{ path: 'c.txt', status: 'conflicted', staged: false }]
    })

    await stageAll()

    expect(api.stage).toHaveBeenCalledWith(REPO, ['a.txt', 'b.txt', 'c.txt'])
  })

  it('commit forwards the message and refreshes', async () => {
    const api = installGitApi({ status: vi.fn().mockResolvedValue(statusPayload()) })

    const ok = await commit('my message')

    expect(ok).toBe(true)
    expect(api.commit).toHaveBeenCalledWith(REPO, 'my message', undefined)
  })

  it('surfaces a failed mutation error', async () => {
    installGitApi({
      status: vi.fn().mockResolvedValue(statusPayload()),
      unstage: vi.fn().mockResolvedValue({ ok: false, error: 'nothing to unstage' })
    })

    const ok = await unstageFiles(['a.txt'])

    expect(ok).toBe(false)
    expect($gitError.get()).toBe('nothing to unstage')
  })
})

describe('branches', () => {
  it('refreshBranches populates local + remote', async () => {
    installGitApi({
      branches: vi.fn().mockResolvedValue({
        ok: true,
        current: 'main',
        local: [{ name: 'main', current: true, ahead: 0, behind: 0, sha: 'abc', subject: 's' }],
        remote: [{ name: 'origin/main', current: false, ahead: 0, behind: 0, sha: 'abc', subject: 's' }]
      })
    })

    await refreshBranches()

    expect($gitBranches.get().local.map(b => b.name)).toEqual(['main'])
    expect($gitBranches.get().remote.map(b => b.name)).toEqual(['origin/main'])
  })

  it('checkoutBranch calls the api and refreshes branches', async () => {
    const api = installGitApi({ status: vi.fn().mockResolvedValue(statusPayload()) })

    const ok = await checkoutBranch('dev')

    expect(ok).toBe(true)
    expect(api.checkout).toHaveBeenCalledWith(REPO, 'dev')
    expect(api.branches).toHaveBeenCalled()
  })

  it('createBranch forwards startPoint when given', async () => {
    const api = installGitApi({ status: vi.fn().mockResolvedValue(statusPayload()) })

    await createBranch('feature/x', 'main')

    expect(api.createBranch).toHaveBeenCalledWith(REPO, 'feature/x', { startPoint: 'main' })
  })
})

describe('log + stash + hunk', () => {
  it('refreshLog populates commits', async () => {
    installGitApi({
      log: vi.fn().mockResolvedValue({
        ok: true,
        commits: [{ sha: 'a', shortSha: 'a', author: 'me', authorEmail: 'e', date: 'd', relativeDate: 'now', subject: 'first' }]
      })
    })

    await refreshLog()

    expect($gitLog.get().map(c => c.subject)).toEqual(['first'])
  })

  it('refreshStashes populates stashes', async () => {
    installGitApi({
      stashList: vi.fn().mockResolvedValue({ ok: true, stashes: [{ ref: 'stash@{0}', subject: 'wip' }] })
    })

    await refreshStashes()

    expect($gitStashes.get().map(s => s.ref)).toEqual(['stash@{0}'])
  })

  it('stashAction forwards action + ref', async () => {
    const api = installGitApi({ status: vi.fn().mockResolvedValue(statusPayload()) })

    await stashAction('pop', 'stash@{0}')

    expect(api.stashAction).toHaveBeenCalledWith(REPO, 'pop', 'stash@{0}')
  })

  it('applyHunk forwards the patch + reverse flag', async () => {
    const api = installGitApi({ status: vi.fn().mockResolvedValue(statusPayload()) })

    await applyHunk('@@ patch @@', true)

    expect(api.applyHunk).toHaveBeenCalledWith(REPO, '@@ patch @@', { reverse: true })
  })
})
