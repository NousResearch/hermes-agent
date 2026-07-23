import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitWorktree, HermesRepoStatus } from '@/global'

import { $repoChangeByPath, $repoStatus, $repoStatusLoading, $repoWorktrees, refreshRepoStatus } from './coding-status'
import { $connection, $currentCwd, $selectedStoredSessionId } from './session'

const sampleStatus: HermesRepoStatus = {
  branch: 'feature/login',
  defaultBranch: 'main',
  detached: false,
  ahead: 1,
  behind: 0,
  staged: 1,
  unstaged: 2,
  untracked: 0,
  conflicted: 0,
  changed: 3,
  added: 12,
  removed: 4,
  files: []
}

function stubProbe(impl: (cwd: string) => Promise<HermesRepoStatus | null>) {
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { git: { repoStatus: impl } }
}

describe('refreshRepoStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $connection.set(null)
    $repoStatus.set(null)
    $repoWorktrees.set([])
    $currentCwd.set('')
    $selectedStoredSessionId.set(null)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllTimers()
    vi.useRealTimers()
    $connection.set(null)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('populates $repoStatus from the probe for an explicit cwd', async () => {
    stubProbe(async () => sampleStatus)
    await refreshRepoStatus('/repo')
    expect($repoStatus.get()).toEqual(sampleStatus)
  })

  it('falls back to the active session cwd when none is passed', async () => {
    const probe = vi.fn(async () => sampleStatus)
    stubProbe(probe)
    $currentCwd.set('/active/repo')
    await refreshRepoStatus()
    expect(probe).toHaveBeenCalledWith('/active/repo')
  })

  it('clears status when there is no cwd', async () => {
    stubProbe(async () => sampleStatus)
    $repoStatus.set(sampleStatus)
    await refreshRepoStatus('   ')
    expect($repoStatus.get()).toBeNull()
  })

  it('clears status when the probe is unavailable (remote backend)', async () => {
    $repoStatus.set(sampleStatus)
    await refreshRepoStatus('/repo')
    expect($repoStatus.get()).toBeNull()
  })

  it('clears status when the probe throws', async () => {
    stubProbe(async () => {
      throw new Error('not a repo')
    })
    $repoStatus.set(sampleStatus)
    await refreshRepoStatus('/repo')
    expect($repoStatus.get()).toBeNull()
  })

  it('clears status, worktrees, and changed-path tints immediately when the cwd changes', () => {
    const status = {
      ...sampleStatus,
      branch: 'feature/a',
      files: [{ conflicted: false, path: 'src/a.ts', staged: false, unstaged: true, untracked: false }]
    }

    const worktrees: HermesGitWorktree[] = [
      { branch: 'feature/a', detached: false, isMain: true, locked: false, path: '/repo-a' }
    ]

    $currentCwd.set('/repo-a')
    $repoStatus.set(status)
    $repoWorktrees.set(worktrees)

    expect($repoChangeByPath.get().get('/repo-a/src/a.ts')).toBe('modified')

    $currentCwd.set('/repo-b')

    expect($repoStatus.get()).toBeNull()
    expect($repoWorktrees.get()).toEqual([])
    expect($repoChangeByPath.get()).toEqual(new Map())
  })

  it('drops a status result that resolves after its cwd becomes inactive', async () => {
    let resolveStatus: (status: HermesRepoStatus | null) => void = () => undefined

    const status = new Promise<HermesRepoStatus | null>(resolve => {
      resolveStatus = resolve
    })

    const worktreeList = vi.fn()

    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
      git: { repoStatus: vi.fn(() => status), worktreeList }
    }
    $currentCwd.set('/repo-a')
    const first = refreshRepoStatus('/repo-a')
    $currentCwd.set('/repo-b')

    resolveStatus({ ...sampleStatus, branch: 'feature/a' })
    await first

    expect($repoStatus.get()).toBeNull()
    expect($repoWorktrees.get()).toEqual([])
    expect(worktreeList).not.toHaveBeenCalled()
  })

  it('drops a worktree result that resolves after its cwd becomes inactive', async () => {
    let resolveWorktrees: (worktrees: HermesGitWorktree[]) => void = () => undefined

    const worktrees = new Promise<HermesGitWorktree[]>(resolve => {
      resolveWorktrees = resolve
    })

    const worktreeList = vi.fn(() => worktrees)

    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
      git: { repoStatus: vi.fn(async () => ({ ...sampleStatus, branch: 'feature/a' })), worktreeList }
    }
    $currentCwd.set('/repo-a')
    const first = refreshRepoStatus('/repo-a')

    await Promise.resolve()
    await Promise.resolve()
    expect(worktreeList).toHaveBeenCalledWith('/repo-a')

    $currentCwd.set('/repo-b')
    resolveWorktrees([{ branch: 'feature/a', detached: false, isMain: true, locked: false, path: '/repo-a' }])
    await first
    await Promise.resolve()

    expect($repoStatus.get()).toBeNull()
    expect($repoWorktrees.get()).toEqual([])
  })

  it('clears immediately when backend identity changes at the same cwd', () => {
    $currentCwd.set('/workspace')
    $connection.set({ baseUrl: 'http://profile-a.test', mode: 'local', profile: 'profile-a' } as never)
    $repoStatus.set({ ...sampleStatus, branch: 'feature/a' })
    $repoWorktrees.set([{ branch: 'feature/a', detached: false, isMain: true, locked: false, path: '/workspace' }])

    $connection.set({ baseUrl: 'http://profile-b.test', mode: 'local', profile: 'profile-b' } as never)

    expect($repoStatus.get()).toBeNull()
    expect($repoWorktrees.get()).toEqual([])
  })

  it('retains a published row during a same-context refresh', async () => {
    let resolveRefresh: (status: HermesRepoStatus | null) => void = () => undefined

    const refresh = new Promise<HermesRepoStatus | null>(resolve => {
      resolveRefresh = resolve
    })

    stubProbe(vi.fn(() => refresh))
    $currentCwd.set('/repo')
    $repoStatus.set({ ...sampleStatus, branch: 'feature/a' })

    const next = refreshRepoStatus('/repo')

    expect($repoStatus.get()).toMatchObject({ branch: 'feature/a' })

    resolveRefresh({ ...sampleStatus, branch: 'feature/b' })
    await next

    expect($repoStatus.get()).toMatchObject({ branch: 'feature/b' })
  })

  it('clears immediately when the stored session changes at the same cwd', () => {
    $currentCwd.set('/repo')
    $selectedStoredSessionId.set('session-a')
    $repoStatus.set({ ...sampleStatus, branch: 'feature/a' })
    $repoWorktrees.set([{ branch: 'feature/a', detached: false, isMain: true, locked: false, path: '/repo' }])

    $selectedStoredSessionId.set('session-b')

    expect($repoStatus.get()).toBeNull()
    expect($repoWorktrees.get()).toEqual([])
  })

  it('runs one probe at a time and coalesces overlap into one trailing refresh', async () => {
    const resolvers: Array<(status: HermesRepoStatus | null) => void> = []
    const calls: string[] = []
    let active = 0
    let maxActive = 0

    stubProbe(
      cwd =>
        new Promise(resolve => {
          calls.push(cwd)
          active++
          maxActive = Math.max(maxActive, active)
          resolvers.push(status => {
            active--
            resolve(status)
          })
        })
    )

    const first = refreshRepoStatus('/repo-a')
    const second = refreshRepoStatus('/repo-b')
    const third = refreshRepoStatus('/repo-c')


    expect(calls).toEqual(['/repo-a'])
    expect(maxActive).toBe(1)
    expect($repoStatusLoading.get()).toBe(true)

    resolvers.shift()?.(sampleStatus)
    await Promise.resolve()
    await Promise.resolve()

    expect(calls).toEqual(['/repo-a', '/repo-c'])
    expect(maxActive).toBe(1)
    expect($repoStatus.get()).toBeNull()

    resolvers.shift()?.(sampleStatus)
    await Promise.all([first, second, third])

    expect(maxActive).toBe(1)
    expect($repoStatus.get()).toEqual(sampleStatus)
    expect($repoStatusLoading.get()).toBe(false)
  })

  it('ignores an A debounce callback that runs after B takes ownership', async () => {
    const probe = vi.fn(async cwd => ({ ...sampleStatus, branch: cwd === '/repo-b' ? 'feature/b' : 'feature/a' }))

    stubProbe(probe)
    vi.clearAllTimers()
    vi.stubGlobal('clearTimeout', () => undefined)

    $currentCwd.set('/repo-a')
    $currentCwd.set('/repo-b')
    vi.advanceTimersByTime(100)
    await vi.runAllTicks()

    expect(probe).toHaveBeenCalledOnce()
    expect(probe).toHaveBeenCalledWith('/repo-b')
    expect($repoStatus.get()).toMatchObject({ branch: 'feature/b' })
  })

  it('refreshes when the stored session id changes even if the cwd is unchanged', async () => {
    const probe = vi.fn(async () => sampleStatus)
    stubProbe(probe)

    $currentCwd.set('/repo')
    $selectedStoredSessionId.set('session-a')
    // The cwd subscription fires on the set above; drain the debounced refresh.
    vi.advanceTimersByTime(200)
    await vi.runAllTicks()

    probe.mockClear()

    // Switch to a different session in the SAME repo dir. The cwd atom value is
    // identical, so its subscription would not re-fire — but the stored-session
    // id did change, which must still trigger a probe so the branch label
    // tracks the new session's checked-out branch.
    $selectedStoredSessionId.set('session-b')
    vi.advanceTimersByTime(200)
    await vi.runAllTicks()

    expect(probe).toHaveBeenCalledWith('/repo')
  })
})
