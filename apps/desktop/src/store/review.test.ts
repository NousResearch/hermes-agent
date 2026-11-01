import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { HermesReviewFile, HermesReviewShipInfo } from '@/global'

import {
  $reviewCommitDefault,
  $reviewCommitMsgBusy,
  $reviewDiff,
  $reviewDiffLoading,
  $reviewFiles,
  $reviewIsRepo,
  $reviewLoading,
  $reviewMaxChurn,
  $reviewOpen,
  $reviewRevertTarget,
  $reviewScope,
  $reviewScopeCwd,
  $reviewScopeTarget,
  $reviewSelectedPath,
  $reviewShipBusy,
  $reviewShipInfo,
  $reviewTurnBase,
  clearReviewSelection,
  closeReview,
  commitChanges,
  confirmRevert,
  createOrOpenPr,
  generateCommitMessage,
  openReview,
  openReviewForPath,
  refreshReview,
  refreshShipInfo,
  requestRevert,
  revealReview,
  revertReviewFile,
  selectReviewFile,
  stageReviewFile,
  toggleReview,
  unstageReviewFile
} from './review'
import { $busy, $currentCwd } from './session'
import { $sessionStates } from './session-states'

// requestOneShot is the only cross-module dependency that must be faked (it
// reaches the gateway); everything else routes through window.hermesDesktop.git,
// which we stub per-test like the sibling coding-status.test.ts does.
const requestOneShot = vi.fn(async (_args: unknown) => 'generated message')
vi.mock('@/lib/oneshot', () => ({ requestOneShot: (args: unknown) => requestOneShot(args) }))
// refreshRepoStatus is a fire-and-forget side effect of mutations; stub it so it
// doesn't try to hit the (absent) probe and log. repoStatusForCwd is read when a
// new PR binds its session to the branch it came from — no probe here, so no
// branch either.
vi.mock('./coding-status', () => ({ refreshRepoStatus: vi.fn(), repoStatusForCwd: () => ({ get: () => null }) }))

function file(path: string, over: Partial<HermesReviewFile> = {}): HermesReviewFile {
  return { path, status: 'modified', staged: false, added: 1, removed: 0, ...over } as HermesReviewFile
}

function deferred<T>() {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

type ReviewStub = Record<string, ReturnType<typeof vi.fn>>

// Install a review bridge on window.hermesDesktop. Any op not supplied defaults
// to a resolved no-op so a test only declares what it exercises.
function stubReview(over: ReviewStub = {}) {
  const review: ReviewStub = {
    list: vi.fn(async () => ({ files: [] })),
    diff: vi.fn(async () => ''),
    stage: vi.fn(async () => undefined),
    unstage: vi.fn(async () => undefined),
    revert: vi.fn(async () => undefined),
    revParse: vi.fn(async () => null),
    commit: vi.fn(async () => undefined),
    commitContext: vi.fn(async () => ({ diff: 'd', recent: 'r' })),
    push: vi.fn(async () => undefined),
    shipInfo: vi.fn(async () => ({ ghReady: false, pr: null })),
    createPr: vi.fn(async () => ({ url: 'https://example.com/pr/1' })),
    ...over
  }

  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
    git: { review },
    openExternal: vi.fn()
  }

  return review
}

beforeEach(() => {
  requestOneShot.mockClear()
  requestOneShot.mockResolvedValue('generated message')
  // Reset stores touched across tests.
  $reviewOpen.set(false)
  $reviewFiles.set([])
  $reviewLoading.set(false)
  $reviewIsRepo.set(true)
  $reviewDiff.set(null)
  $reviewDiffLoading.set(false)
  $reviewSelectedPath.set(null)
  $reviewShipInfo.set({ ghReady: false, pr: null })
  $reviewShipBusy.set(false)
  $reviewCommitMsgBusy.set(false)
  $reviewRevertTarget.set(undefined)
  $reviewScope.set('uncommitted')
  $reviewTurnBase.set({})
  $reviewScopeCwd.set(null)
  $reviewScopeTarget.set('main')
  $currentCwd.set('/repo')
  $busy.set(false)
  $sessionStates.set({})
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('refreshReview', () => {
  it('is a no-op that clears state when the pane is closed', async () => {
    const review = stubReview()
    $reviewOpen.set(false)
    $reviewFiles.set([file('a.ts')])

    await refreshReview()

    expect(review.list).not.toHaveBeenCalled()
    expect($reviewFiles.get()).toEqual([])
    expect($reviewLoading.get()).toBe(false)
  })

  it('flags not-a-repo (and clears loading) when there is no bridge/cwd', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    $reviewOpen.set(true)
    $reviewLoading.set(true)

    await refreshReview()

    expect($reviewIsRepo.get()).toBe(false)
    expect($reviewLoading.get()).toBe(false)
  })

  it('populates the changed-file list from the bridge', async () => {
    stubReview({ list: vi.fn(async () => ({ files: [file('a.ts'), file('b.ts')] })) })
    $reviewOpen.set(true)

    await refreshReview()

    expect($reviewFiles.get().map(f => f.path)).toEqual(['a.ts', 'b.ts'])
    expect($reviewIsRepo.get()).toBe(true)
    expect($reviewLoading.get()).toBe(false)
  })

  it('filters excluded paths (node_modules et al.) out of the list', async () => {
    stubReview({ list: vi.fn(async () => ({ files: [file('src/a.ts'), file('node_modules/x/index.js')] })) })
    $reviewOpen.set(true)

    await refreshReview()

    expect($reviewFiles.get().map(f => f.path)).toEqual(['src/a.ts'])
  })

  it('drops a selection whose file vanished from the new list', async () => {
    stubReview({ list: vi.fn(async () => ({ files: [file('kept.ts')] })) })
    $reviewOpen.set(true)
    $reviewSelectedPath.set('gone.ts')
    $reviewDiff.set('old diff')

    await refreshReview()

    expect($reviewSelectedPath.get()).toBeNull()
    expect($reviewDiff.get()).toBeNull()
  })

  it('clears the list but keeps isRepo true when the bridge throws', async () => {
    stubReview({
      list: vi.fn(async () => {
        throw new Error('git failed')
      })
    })
    $reviewOpen.set(true)
    $reviewFiles.set([file('stale.ts')])

    await refreshReview()

    expect($reviewFiles.get()).toEqual([])
    expect($reviewIsRepo.get()).toBe(true)
    expect($reviewLoading.get()).toBe(false)
  })

  it('keeps a new repository loading when the previous request rejects during the debounce gap', async () => {
    vi.useFakeTimers()

    const repoA = deferred<{ files: HermesReviewFile[] }>()
    const repoB = deferred<{ files: HermesReviewFile[] }>()

    const review = stubReview({
      list: vi.fn((cwd: string) => (cwd === '/repo-a' ? repoA.promise : repoB.promise))
    })

    $reviewOpen.set(true)
    $currentCwd.set('/repo-a')

    const staleRefresh = refreshReview()
    expect($reviewLoading.get()).toBe(true)

    $currentCwd.set('/repo-b')
    expect($reviewLoading.get()).toBe(true)

    repoA.reject(new Error('repo A disappeared'))
    await staleRefresh

    expect($reviewLoading.get()).toBe(true)
    expect($reviewFiles.get()).toEqual([])

    await vi.advanceTimersByTimeAsync(100)
    expect(review.list).toHaveBeenLastCalledWith('/repo-b', 'uncommitted', null)

    repoB.resolve({ files: [file('b.ts')] })
    await vi.runAllTimersAsync()
    await Promise.resolve()

    expect($reviewFiles.get().map(entry => entry.path)).toEqual(['b.ts'])
    expect($reviewLoading.get()).toBe(false)
  })

  it('does not let an older list response clear a newer direct selection', async () => {
    const pendingList = deferred<{ files: HermesReviewFile[] }>()
    stubReview({ list: vi.fn(() => pendingList.promise), diff: vi.fn(async () => 'new diff') })
    $reviewOpen.set(true)

    const staleRefresh = refreshReview()
    await selectReviewFile(file('b.ts'))

    pendingList.resolve({ files: [file('a.ts')] })
    await staleRefresh

    expect($reviewSelectedPath.get()).toBe('b.ts')
    expect($reviewDiff.get()).toBe('new diff')
  })

  it('does not let an older finally clear a newer in-flight refresh spinner', async () => {
    const first = deferred<{ files: HermesReviewFile[] }>()
    const second = deferred<{ files: HermesReviewFile[] }>()
    let call = 0
    stubReview({ list: vi.fn(() => (++call === 1 ? first.promise : second.promise)) })
    $reviewOpen.set(true)

    const staleRefresh = refreshReview()
    const currentRefresh = refreshReview()

    first.reject(new Error('older request failed'))
    await staleRefresh
    expect($reviewLoading.get()).toBe(true)

    second.resolve({ files: [file('current.ts')] })
    await currentRefresh

    expect($reviewFiles.get().map(entry => entry.path)).toEqual(['current.ts'])
    expect($reviewLoading.get()).toBe(false)
  })
})

describe('$reviewMaxChurn', () => {
  it('is the largest added+removed across files', () => {
    $reviewFiles.set([file('a', { added: 3, removed: 2 }), file('b', { added: 10, removed: 1 }), file('c')])
    expect($reviewMaxChurn.get()).toBe(11)
  })

  it('is 0 for an empty list', () => {
    $reviewFiles.set([])
    expect($reviewMaxChurn.get()).toBe(0)
  })
})

describe('selectReviewFile', () => {
  it('sets the selected path and fetches its diff', async () => {
    const review = stubReview({ diff: vi.fn(async () => 'the diff') })

    await selectReviewFile(file('a.ts'))

    expect($reviewSelectedPath.get()).toBe('a.ts')
    expect($reviewDiff.get()).toBe('the diff')
    expect($reviewDiffLoading.get()).toBe(false)
    expect(review.diff).toHaveBeenCalledWith('/repo', 'a.ts', 'uncommitted', null, false)
  })

  it('fetches the diff for the selected scope and base', async () => {
    const review = stubReview({ diff: vi.fn(async () => 'd') })
    $reviewScope.set('branch')

    await selectReviewFile(file('a.ts'))
    expect(review.diff).toHaveBeenCalledWith('/repo', 'a.ts', 'branch', null, false)

    $reviewScope.set('lastTurn')
    $reviewTurnBase.set({ '/repo': 'abc123' })
    await selectReviewFile(file('a.ts'))
    expect(review.diff).toHaveBeenCalledWith('/repo', 'a.ts', 'lastTurn', 'abc123', false)
  })

  it('coerces a falsy diff to empty string (not null)', async () => {
    stubReview({ diff: vi.fn(async () => '') })

    await selectReviewFile(file('a.ts'))

    expect($reviewDiff.get()).toBe('')
  })

  it('sets diff null when there is no bridge', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop

    await selectReviewFile(file('a.ts'))

    expect($reviewSelectedPath.get()).toBe('a.ts')
    expect($reviewDiff.get()).toBeNull()
  })

  it('does not let an older same-path diff overwrite a newer selection request', async () => {
    const first = deferred<string>()
    const second = deferred<string>()
    let call = 0
    stubReview({ diff: vi.fn(() => (++call === 1 ? first.promise : second.promise)) })

    const staleSelection = selectReviewFile(file('a.ts'))
    const currentSelection = selectReviewFile(file('a.ts'))

    second.resolve('current diff')
    await currentSelection
    first.resolve('stale diff')
    await staleSelection

    expect($reviewSelectedPath.get()).toBe('a.ts')
    expect($reviewDiff.get()).toBe('current diff')
    expect($reviewDiffLoading.get()).toBe(false)
  })

  it('clears path, diff and loading', () => {
    $reviewSelectedPath.set('a.ts')
    $reviewDiff.set('x')
    $reviewDiffLoading.set(true)

    clearReviewSelection()

    expect($reviewSelectedPath.get()).toBeNull()
    expect($reviewDiff.get()).toBeNull()
    expect($reviewDiffLoading.get()).toBe(false)
  })
})

describe('view state', () => {
  it('openReview opens the pane and kicks off a refresh', async () => {
    const review = stubReview()
    openReview()
    expect($reviewOpen.get()).toBe(true)
    expect($reviewScopeCwd.get()).toBeNull()
    // openReview fires refreshReview + refreshShipInfo without awaiting.
    await Promise.resolve()
    await Promise.resolve()
    expect(review.list).toHaveBeenCalledWith('/repo', 'uncommitted', null)
  })

  it('openReview pins the pane to a tile worktree when scoped', async () => {
    const review = stubReview({
      list: vi.fn(async () => ({ files: [file('tile.ts')] }))
    })

    openReview('/tile-worktree')
    expect($reviewOpen.get()).toBe(true)
    expect($reviewScopeCwd.get()).toBe('/tile-worktree')
    await Promise.resolve()
    await Promise.resolve()
    expect(review.list).toHaveBeenCalledWith('/tile-worktree', 'uncommitted', null)
  })

  it('revealReview re-homes the origin when the repo stays the same', () => {
    stubReview()
    openReview('/tile-worktree', 'tile:project-a')

    revealReview('/tile-worktree', 'tile:project-b')

    expect($reviewScopeTarget.get()).toBe('tile:project-b')
  })

  it('narrow toggle re-homes the origin before showing the overlay', () => {
    const originalMatchMedia = window.matchMedia

    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: vi.fn(() => ({ matches: true }))
    })

    try {
      stubReview()
      openReview('/project-a', 'tile:project-a')

      toggleReview('/project-b', 'tile:project-b')

      expect($reviewScopeCwd.get()).toBe('/project-b')
      expect($reviewScopeTarget.get()).toBe('tile:project-b')
    } finally {
      Object.defineProperty(window, 'matchMedia', { configurable: true, value: originalMatchMedia })
    }
  })

  it('closeReview closes the pane, clears selection, and drops scope', () => {
    stubReview()
    $reviewOpen.set(true)
    $reviewScopeCwd.set('/tile-worktree')
    $reviewSelectedPath.set('a.ts')
    $reviewDiff.set('x')

    closeReview()

    expect($reviewOpen.get()).toBe(false)
    expect($reviewScopeCwd.get()).toBeNull()
    expect($reviewScopeTarget.get()).toBe('main')
    expect($reviewSelectedPath.get()).toBeNull()
    expect($reviewDiff.get()).toBeNull()
  })

  it('scoped pane ignores main-pane cwd changes', async () => {
    const review = stubReview({
      list: vi.fn(async (cwd: string) => ({ files: [file(cwd === '/tile' ? 'tile.ts' : 'main.ts')] }))
    })

    openReview('/tile')
    await Promise.resolve()
    await Promise.resolve()
    review.list.mockClear()

    // Main session hops repos; the pane is still pinned to the tile.
    $currentCwd.set('/somewhere-else')
    await Promise.resolve()
    await Promise.resolve()

    expect($reviewScopeCwd.get()).toBe('/tile')
    expect(review.list).not.toHaveBeenCalled()
  })

  it('keeps openReviewForPath ownership when a debounced refresh was already pending', async () => {
    vi.useFakeTimers()
    const directList = deferred<{ files: HermesReviewFile[] }>()
    const review = stubReview({ list: vi.fn(() => directList.promise), diff: vi.fn(async () => 'target diff') })
    $reviewOpen.set(true)

    // A repository move arms the debounce before the direct file-open intent.
    $currentCwd.set('/repo-next')
    const openTarget = openReviewForPath('/repo-next/target.ts', '/repo-next', 'tile:project-b')

    expect(review.list).toHaveBeenCalledTimes(1)
    expect($reviewScopeTarget.get()).toBe('tile:project-b')
    await vi.advanceTimersByTimeAsync(100)

    directList.resolve({ files: [file('target.ts')] })
    await openTarget

    expect(review.list).toHaveBeenCalledTimes(1)
    expect($reviewSelectedPath.get()).toBe('target.ts')
    expect($reviewDiff.get()).toBe('target diff')
  })

  it('openReviewForPath falls through to another scope when the pane\'s scope cannot list the clicked file', async () => {
    // Restart wiped the in-memory lastTurn baseline: the pane's persisted scope
    // lists nothing (base null → empty), but the file exists in 'branch'.
    const review = stubReview({
      list: vi.fn(async (_cwd: string, scope: string) =>
        scope === 'branch' ? { files: [file('Service.ts')] } : { files: [] }
      ),
      diff: vi.fn(async () => 'branch diff')
    })

    $reviewOpen.set(true)
    $reviewScope.set('lastTurn')

    await openReviewForPath('Service.ts')

    expect($reviewScope.get()).toBe('branch')
    expect($reviewSelectedPath.get()).toBe('Service.ts')
    expect($reviewDiff.get()).toBe('branch diff')
    expect(review.list).toHaveBeenLastCalledWith('/repo', 'branch', null)
  })

  it('openReviewForPath restores the user\'s scope when no scope lists the clicked file', async () => {
    const review = stubReview({ list: vi.fn(async () => ({ files: [file('other.ts')] })) })
    $reviewOpen.set(true)
    $reviewScope.set('lastTurn')

    await openReviewForPath('vanished.ts')

    // The card was stale; the pane is left exactly as the user had it.
    expect($reviewScope.get()).toBe('lastTurn')
    expect($reviewSelectedPath.get()).toBeNull()
    // Initial list + one per alternative scope + the restoring refresh.
    expect(review.list).toHaveBeenCalledTimes(4)
  })
})

describe('mutations', () => {
  it('stageReviewFile forwards the path and re-syncs', async () => {
    const review = stubReview()
    $reviewOpen.set(true) // afterMutation's refreshReview only lists when the pane is open
    await stageReviewFile('a.ts')
    expect(review.stage).toHaveBeenCalledWith('/repo', 'a.ts')
    expect(review.list).toHaveBeenCalled()
  })
})

describe('revert confirm dialog', () => {
  it('requestRevert(null) encodes the "revert all" target distinctly from closed', () => {
    requestRevert(null)
    expect($reviewRevertTarget.get()).toEqual({ path: null })
  })

  it('confirmRevert closes the dialog then performs the revert', async () => {
    const review = stubReview()
    requestRevert('a.ts')

    await confirmRevert()

    expect($reviewRevertTarget.get()).toBeUndefined()
    expect(review.revert).toHaveBeenCalledWith('/repo', 'a.ts')
  })

  it('confirmRevert is a no-op when nothing is pending', async () => {
    const review = stubReview()
    $reviewRevertTarget.set(undefined)

    await confirmRevert()

    expect(review.revert).not.toHaveBeenCalled()
  })
})

describe('ship flow', () => {
  it('commitChanges commits the trimmed message and toggles the busy flag', async () => {
    const review = stubReview()
    const seen: boolean[] = []
    const unsub = $reviewShipBusy.subscribe(v => seen.push(v))

    await commitChanges('  a message  ', { push: true })

    expect(review.commit).toHaveBeenCalledWith('/repo', 'a message', true)
    expect(seen).toContain(true)
    expect($reviewShipBusy.get()).toBe(false)
    unsub()
  })

  it('commitChanges bails on a blank message', async () => {
    const review = stubReview()
    await commitChanges('   ')
    expect(review.commit).not.toHaveBeenCalled()
  })

  it('createOrOpenPr opens the existing PR without creating a new one', async () => {
    const review = stubReview()
    $reviewShipInfo.set({ ghReady: true, pr: { url: 'https://example.com/pr/9' } } as HermesReviewShipInfo)

    await createOrOpenPr()

    expect(review.createPr).not.toHaveBeenCalled()
    expect(
      (window.hermesDesktop as unknown as { openExternal: ReturnType<typeof vi.fn> }).openExternal
    ).toHaveBeenCalledWith('https://example.com/pr/9')
  })

  it('createOrOpenPr creates a PR when none exists, then opens it', async () => {
    const review = stubReview({ createPr: vi.fn(async () => ({ url: 'https://example.com/pr/new' })) })
    $reviewShipInfo.set({ ghReady: true, pr: null })

    await createOrOpenPr()

    expect(review.createPr).toHaveBeenCalledWith('/repo')
    expect(
      (window.hermesDesktop as unknown as { openExternal: ReturnType<typeof vi.fn> }).openExternal
    ).toHaveBeenCalledWith('https://example.com/pr/new')
  })
})

describe('refreshShipInfo', () => {
  it('resets ship info when there is no bridge', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    $reviewShipInfo.set({ ghReady: true, pr: { url: 'x' } } as HermesReviewShipInfo)

    await refreshShipInfo()

    expect($reviewShipInfo.get()).toEqual({ ghReady: false, pr: null })
  })

  it('resets ship info when the bridge throws', async () => {
    stubReview({
      shipInfo: vi.fn(async () => {
        throw new Error('gh missing')
      })
    })
    $reviewShipInfo.set({ ghReady: true, pr: { url: 'x' } } as HermesReviewShipInfo)

    await refreshShipInfo()

    expect($reviewShipInfo.get()).toEqual({ ghReady: false, pr: null })
  })
})

describe('generateCommitMessage', () => {
  it('returns a one-shot message from the working-tree diff', async () => {
    stubReview()

    const msg = await generateCommitMessage('avoid this')

    expect(msg).toBe('generated message')
    expect(requestOneShot).toHaveBeenCalledWith(
      expect.objectContaining({
        template: 'commit_message',
        variables: expect.objectContaining({ avoid: 'avoid this', diff: 'd', recent_commits: 'r' })
      })
    )
    expect($reviewCommitMsgBusy.get()).toBe(false)
  })

  it('returns empty (no model call) when the diff is blank', async () => {
    stubReview({ commitContext: vi.fn(async () => ({ diff: '   ', recent: '' })) })

    const msg = await generateCommitMessage()

    expect(msg).toBe('')
    expect(requestOneShot).not.toHaveBeenCalled()
  })

  it('returns empty when the bridge lacks commitContext', async () => {
    const review = stubReview()
    delete review.commitContext

    const msg = await generateCommitMessage()

    expect(msg).toBe('')
  })
})

describe('$reviewCommitDefault', () => {
  it('remembers the split-button default action', () => {
    $reviewCommitDefault.set('commitPush')
    expect($reviewCommitDefault.get()).toBe('commitPush')
    $reviewCommitDefault.set('commit')
    expect($reviewCommitDefault.get()).toBe('commit')
  })
})

describe('$reviewScope', () => {
  it('round-trips the three diff scopes', () => {
    expect($reviewScope.get()).toBe('uncommitted')
    $reviewScope.set('branch')
    expect($reviewScope.get()).toBe('branch')
    $reviewScope.set('lastTurn')
    expect($reviewScope.get()).toBe('lastTurn')
    $reviewScope.set('uncommitted')
    expect($reviewScope.get()).toBe('uncommitted')
  })
})

describe('$reviewTurnBase', () => {
  it('maps each repo cwd to its last-turn HEAD baseline', () => {
    expect($reviewTurnBase.get()).toEqual({})
    $reviewTurnBase.set({ '/repo': 'abc123' })
    expect($reviewTurnBase.get()).toEqual({ '/repo': 'abc123' })
  })

  it('caps tracked baselines at MAX_TURN_BASES, evicting the least-recently captured', async () => {
    stubReview({ revParse: vi.fn(async (cwd: string) => `sha-${cwd}`) })

    // Start with the map already full of older baselines.
    $reviewTurnBase.set(Object.fromEntries(Array.from({ length: 8 }, (_, i) => [`/old-${i}`, `sha-${i}`])))

    // A fresh turn in a ninth repo must evict the oldest (/old-0), not grow.
    $sessionStates.set({ rt_cap: sessionState(true, '/new-repo') })

    await new Promise(resolve => setTimeout(resolve, 0))

    const bases = $reviewTurnBase.get()
    expect(Object.keys(bases)).toHaveLength(8)
    expect(bases['/new-repo']).toBe('sha-/new-repo')
    expect(bases['/old-0']).toBeUndefined()
    expect(bases['/old-7']).toBeDefined()

    // Re-capturing an existing cwd refreshes it without growing the map.
    $sessionStates.set({ rt_recapture: sessionState(true, '/old-1') })

    await new Promise(resolve => setTimeout(resolve, 0))

    expect($reviewTurnBase.get()['/old-1']).toBe('sha-/old-1')
    expect(Object.keys($reviewTurnBase.get())).toHaveLength(8)
  })
})

// Minimal session state: the baseline capture only reads `busy` and `cwd`.
const sessionState = (busy: boolean, cwd = '/repo'): ClientSessionState => ({ busy, cwd }) as ClientSessionState

describe('turn baseline capture', () => {
  it('captures HEAD per-cwd when a session turn starts', async () => {
    stubReview({ revParse: vi.fn(async () => 'deadbeef') })

    $sessionStates.set({ rt_capture: sessionState(true, '/repo') })

    await new Promise(resolve => setTimeout(resolve, 0))
    expect($reviewTurnBase.get()).toEqual({ '/repo': 'deadbeef' })
  })

  it('keys baselines by cwd so background / tile worktrees get their own', async () => {
    stubReview({ revParse: vi.fn(async (cwd: string) => (cwd === '/tile' ? 'tile-sha' : 'main-sha')) })

    $sessionStates.set({ rt_main: sessionState(true, '/repo'), rt_tile: sessionState(true, '/tile') })

    await new Promise(resolve => setTimeout(resolve, 0))
    expect($reviewTurnBase.get()).toEqual({ '/repo': 'main-sha', '/tile': 'tile-sha' })
  })

  it('leaves the baseline empty when there is no git bridge', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop

    $sessionStates.set({ rt_nobridge: sessionState(true, '/repo') })

    await new Promise(resolve => setTimeout(resolve, 0))
    expect($reviewTurnBase.get()).toEqual({})
  })
})

describe('mutation gating', () => {
  it('skips stage/unstage/revert when the scope is not uncommitted', async () => {
    const review = stubReview()
    $reviewScope.set('branch')

    await stageReviewFile('a.ts')
    await unstageReviewFile('a.ts')
    await revertReviewFile('a.ts')

    expect(review.stage).not.toHaveBeenCalled()
    expect(review.unstage).not.toHaveBeenCalled()
    expect(review.revert).not.toHaveBeenCalled()
  })

  it('still mutates under the default uncommitted scope', async () => {
    const review = stubReview()

    await stageReviewFile('a.ts')

    expect(review.stage).toHaveBeenCalledWith('/repo', 'a.ts')
  })
})
