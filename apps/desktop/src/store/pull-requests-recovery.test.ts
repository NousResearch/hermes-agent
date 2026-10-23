/**
 * #136048 — the sidebar PR tag must not miss or freeze a session's PR.
 *
 * `recoverSessionPullRequests` used to scan each session's transcript once,
 * ever: a live session is listed (and so scanned) before the agent has opened
 * its PR, the miss went into the persisted scan list, and the later
 * `gh pr create` output — or a replacement PR — was never looked at. These
 * cases drive the real store module against mocked desktop APIs and pin the
 * fix: the scan result is keyed on the transcript revision it actually saw.
 *
 * Scenario A/B/E/F are RED against the old bookkeeping; C/D/G are the
 * preservation controls (cache, branch join, older-backend bail-out).
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

type ScanResponse = { pull_requests: Record<string, { number: number; url: string }>; scanned: string[] }

type Pr = { branch: string; draft: boolean; number: number; state: string; title: string; url: string }

const scanMock = vi.fn<(ids: string[]) => Promise<ScanResponse>>()

// Only the scan RPC is replaced; the store imports nothing else from `@/hermes`.
vi.mock('@/hermes', () => ({
  scanSessionPullRequests: (ids: string[]) => scanMock(ids)
}))

const prListMock = vi.fn<(root: string, branches: string[], numbers: number[]) => Promise<{ prs: Pr[] }>>()

vi.mock('@/lib/desktop-git', () => ({
  desktopGit: () => ({
    review: {
      prList: (root: string, branches: string[], numbers: number[]) => prListMock(root, branches, numbers)
    }
  })
}))

const REPO = '/work/checkout'

/** A sidebar row. `revision` stands in for the transcript length the scan is
 *  keyed on (`message_count`), and grows when the session produces more. */
const row = (id: string, revision: number, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    ended_at: null,
    git_branch: null,
    git_repo_root: REPO,
    id,
    input_tokens: 0,
    is_active: true,
    last_active: revision,
    message_count: revision,
    model: 'test-model',
    output_tokens: 0,
    ...extra
  }) as SessionInfo

const pr = (number: number): Pr => ({
  branch: 'feat/x',
  draft: false,
  number,
  state: 'open',
  title: `pr ${number}`,
  url: `https://github.com/o/r/pull/${number}`
})

/** The next recovery pass answers with `found` for the ids it asked about. */
const willScan = (found: ScanResponse['pull_requests'], asked: string[]) =>
  scanMock.mockResolvedValueOnce({ pull_requests: found, scanned: asked })

/** The store's atoms (and the persisted bookkeeping) are module-level: a case
 *  that cares about first load gets a fresh copy of the module. */
const loadStore = async () => {
  vi.resetModules()

  return await import('./pull-requests')
}

beforeEach(() => {
  window.localStorage.clear()
  scanMock.mockReset()
  prListMock.mockReset()
})

describe('recoverSessionPullRequests — transcript revision', () => {
  it('A. finds a PR that appeared after the first scan missed it', async () => {
    const store = await loadStore()

    // The session is listed while it is still working: no PR yet, so the scan
    // comes back empty.
    willScan({}, ['s-miss'])
    await store.recoverSessionPullRequests([row('s-miss', 10)])

    expect(scanMock).toHaveBeenCalledTimes(1)
    expect(scanMock).toHaveBeenLastCalledWith(['s-miss'])
    expect(store.$prBranchBySession.get()['s-miss']).toBeUndefined()

    // The agent runs `gh pr create`: the transcript grows, and a later pass
    // must be able to see the PR.
    willScan({ 's-miss': { number: 7, url: pr(7).url } }, ['s-miss'])
    await store.recoverSessionPullRequests([row('s-miss', 20)])

    expect(scanMock).toHaveBeenCalledTimes(2)
    expect(store.$prBranchBySession.get()['s-miss']).toBe(store.numberPrKey(REPO, 7))
  })

  it('B. a later PR replaces the stamp instead of freezing the first one', async () => {
    const store = await loadStore()

    willScan({ 's-replace': { number: 1, url: pr(1).url } }, ['s-replace'])
    await store.recoverSessionPullRequests([row('s-replace', 5)])

    expect(store.$prBranchBySession.get()['s-replace']).toBe(store.numberPrKey(REPO, 1))

    // Same session, second PR (a follow-up, or the replacement after a closed
    // first attempt).
    willScan({ 's-replace': { number: 2, url: pr(2).url } }, ['s-replace'])
    await store.recoverSessionPullRequests([row('s-replace', 30)])

    expect(scanMock).toHaveBeenCalledTimes(2)
    expect(store.$prBranchBySession.get()['s-replace']).toBe(store.numberPrKey(REPO, 2))

    // ...and the replacement is not re-asked while the transcript stands still.
    await store.recoverSessionPullRequests([row('s-replace', 30)])
    expect(scanMock).toHaveBeenCalledTimes(2)
  })

  it('C. an unchanged transcript is never re-asked, hit or miss', async () => {
    const store = await loadStore()

    willScan({}, ['s-miss'])
    await store.recoverSessionPullRequests([row('s-miss', 7)])
    await store.recoverSessionPullRequests([row('s-miss', 7)])

    willScan({ 's-hit': { number: 3, url: pr(3).url } }, ['s-hit'])
    await store.recoverSessionPullRequests([row('s-hit', 4)])
    await store.recoverSessionPullRequests([row('s-hit', 4)])
    await store.recoverSessionPullRequests([row('s-hit', 4)])

    // One scan each — the second pass has no new revision to justify asking.
    expect(scanMock).toHaveBeenCalledTimes(2)
  })

  it('D. the branch join still owns a session on a feature branch', async () => {
    const store = await loadStore()
    const branchRow = row('s-branch', 9, { git_branch: 'feat/live-branch' })

    await store.recoverSessionPullRequests([branchRow])

    // The row answers by branch, so it is never transcript-scanned and never
    // stamped — unchanged from before.
    expect(scanMock).not.toHaveBeenCalled()
    expect(store.$prBranchBySession.get()['s-branch']).toBeUndefined()
    expect(store.sessionPrKey(branchRow)).toBe(store.branchPrKey(REPO, 'feat/live-branch'))

    // A trunk checkout has no branch to join, so it is still scanned.
    willScan({}, ['s-trunk'])
    await store.recoverSessionPullRequests([row('s-trunk', 3, { git_branch: 'main' })])

    expect(scanMock).toHaveBeenCalledTimes(1)
    expect(scanMock).toHaveBeenLastCalledWith(['s-trunk'])
  })

  it('E. the row badge follows the latest recovered PR', async () => {
    const store = await loadStore()

    willScan({ 's-badge': { number: 1, url: pr(1).url } }, ['s-badge'])
    await store.recoverSessionPullRequests([row('s-badge', 5)])

    willScan({ 's-badge': { number: 2, url: pr(2).url } }, ['s-badge'])
    const grew = row('s-badge', 30)
    await store.recoverSessionPullRequests([grew])

    // What the sidebar does with the stamp: split it into repo + lookup and
    // resolve it through `prList`.
    const key = store.sessionPrKey(grew)
    expect(key).not.toBeNull()

    const [root, lookup] = (key as string).split('\n')
    prListMock.mockResolvedValue({ prs: [pr(2)] })
    await store.refreshPullRequests({ [root]: [lookup] })

    expect(prListMock).toHaveBeenCalledWith(root, [], [2])
    expect(store.$pullRequestsByBranch.get()[key as string]?.number).toBe(2)
    expect(store.pullRequestBucket(store.$pullRequestsByBranch.get()[key as string])).toBe('open')
  })

  it('F. revisions persist across a reload; the legacy key re-scans once', async () => {
    window.localStorage.setItem('hermes.desktop.prScanRevisions', JSON.stringify({ 's-known': '12' }))

    let store = await loadStore()
    await store.recoverSessionPullRequests([row('s-known', 12)])
    expect(scanMock).not.toHaveBeenCalled()

    // Growth after a reload still buys exactly one look.
    willScan({}, ['s-known'])
    await store.recoverSessionPullRequests([row('s-known', 13)])

    expect(scanMock).toHaveBeenCalledTimes(1)
    expect(JSON.parse(window.localStorage.getItem('hermes.desktop.prScanRevisions') as string)).toMatchObject({
      's-known': '13'
    })

    // A pre-upgrade build left only the id list. No revision is known for it,
    // so the first pass re-scans once — the accepted one-time upgrade cost.
    window.localStorage.clear()
    window.localStorage.setItem('hermes.desktop.prScannedSessions', JSON.stringify(['s-legacy']))
    scanMock.mockReset()

    store = await loadStore()
    willScan({}, ['s-legacy'])
    await store.recoverSessionPullRequests([row('s-legacy', 4)])
    await store.recoverSessionPullRequests([row('s-legacy', 4)])

    expect(scanMock).toHaveBeenCalledTimes(1)
  })

  it('G. an older backend without the route is still left alone', async () => {
    const store = await loadStore()

    scanMock.mockRejectedValueOnce(new Error('404: Not Found'))
    await store.recoverSessionPullRequests([row('s-old-backend', 1)])
    await store.recoverSessionPullRequests([row('s-old-backend', 2)])

    // `scanUnavailable` latches on the first failure.
    expect(scanMock).toHaveBeenCalledTimes(1)
  })

  it('H. one pass batches every changed session into a single scan', async () => {
    const store = await loadStore()

    willScan({}, ['s-a', 's-b'])
    await store.recoverSessionPullRequests([
      row('s-a', 1),
      row('s-b', 1, { git_branch: 'feat/x' }) // branch join owns this one
    ])

    expect(scanMock).toHaveBeenCalledTimes(1)
    expect(scanMock).toHaveBeenLastCalledWith(['s-a'])

    willScan({}, ['s-a', 's-b'])
    await store.recoverSessionPullRequests([row('s-a', 2), row('s-b', 9)])

    expect(scanMock).toHaveBeenCalledTimes(2)
    expect(scanMock).toHaveBeenLastCalledWith(['s-a', 's-b'])
  })

  it('I. a re-scan that recovers nothing keeps the earlier stamp', async () => {
    const store = await loadStore()

    willScan({ 's-keep': { number: 4, url: pr(4).url } }, ['s-keep'])
    await store.recoverSessionPullRequests([row('s-keep', 6)])

    expect(store.$prBranchBySession.get()['s-keep']).toBe(store.numberPrKey(REPO, 4))

    // The transcript grew, but the PR url fell out of the scanned window: the
    // row must keep pointing at what was already recovered.
    willScan({}, ['s-keep'])
    await store.recoverSessionPullRequests([row('s-keep', 40)])

    expect(store.$prBranchBySession.get()['s-keep']).toBe(store.numberPrKey(REPO, 4))

    // ...and the advanced revision is remembered, so it is not asked again.
    await store.recoverSessionPullRequests([row('s-keep', 40)])
    expect(scanMock).toHaveBeenCalledTimes(2)
  })
})
