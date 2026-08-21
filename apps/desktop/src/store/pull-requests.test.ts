// The PR store's join keys and fetch coordination. The lane badge (workspace
// headers) and the session-row badge share one map and one fetcher; these
// tests pin the key guards (trunk, kanban callers pass nothing) and the two
// fetch behaviors the lane badge added: lookup-coverage staleness (a fresh
// repo still re-fetches for an uncovered branch) and lookup merging (a
// narrow ask must not shrink the repo's slice — the store replaces a repo's
// PRs wholesale).
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesBranchPullRequest } from '@/global'

vi.mock('@/lib/desktop-git', () => ({
  desktopGit: vi.fn()
}))

vi.mock('@/hermes', () => ({
  scanSessionPullRequests: vi.fn()
}))

const git = await import('@/lib/desktop-git')
const desktopGit = vi.mocked(git.desktopGit)

const { $pullRequestsByBranch, branchLanePrKey, branchPrKey, refreshPullRequests } =
  await import('@/store/pull-requests')

const pr = (number: number, branch: string): HermesBranchPullRequest =>
  ({ branch, draft: false, number, state: 'open', title: `pr ${number}`, url: `https://x/${number}` }) as never

const prListMock = (prs: HermesBranchPullRequest[]) => {
  const prList = vi.fn().mockResolvedValue({ prs })

  desktopGit.mockReturnValue({ review: { prList } } as never)

  return prList
}

describe('branchLanePrKey', () => {
  it('builds the same key a session row uses for the branch join', () => {
    expect(branchLanePrKey('/repo', 'feat/x')).toBe(branchPrKey('/repo', 'feat/x'))
  })

  it('returns null for trunk branches — asking GitHub about main badges a stranger fork PR onto it', () => {
    for (const trunk of ['main', 'MASTER', 'develop', 'dev', 'trunk']) {
      expect(branchLanePrKey('/repo', trunk)).toBeNull()
    }
  })

  it('returns null without both halves of the key', () => {
    expect(branchLanePrKey(null, 'feat/x')).toBeNull()
    expect(branchLanePrKey('/repo', null)).toBeNull()
    expect(branchLanePrKey('/repo', '  ')).toBeNull()
    expect(branchLanePrKey(undefined, undefined)).toBeNull()
  })
})

describe('refreshPullRequests', () => {
  beforeEach(() => {
    $pullRequestsByBranch.set({})
    vi.clearAllMocks()
  })

  it('re-fetches a fresh repo when the ask includes an uncovered lookup', async () => {
    const first = prListMock([pr(1, 'feat/a')])

    await refreshPullRequests({ '/repo': ['feat/a'] })
    expect(first).toHaveBeenCalledTimes(1)

    // Same lookups, inside the TTL: stays cached.
    await refreshPullRequests({ '/repo': ['feat/a'] })
    expect(first).toHaveBeenCalledTimes(1)

    // A NEW lookup (a lane appearing) must not be starved by the TTL.
    const second = prListMock([pr(1, 'feat/a'), pr(2, 'feat/b')])

    await refreshPullRequests({ '/repo': ['feat/b'] })
    expect(second).toHaveBeenCalledTimes(1)
  })

  it('merges covered lookups into a narrow ask so the wholesale replace keeps sibling PRs', async () => {
    prListMock([pr(1, 'feat/a'), pr(2, 'feat/b')])
    await refreshPullRequests({ '/repo': ['feat/a', 'feat/b'] })

    // A narrow force-refresh (one lane) must still ask about both branches:
    // the store replaces the repo's slice wholesale, so an uncovered sibling
    // would otherwise vanish from the session rows that render it.
    const narrow = prListMock([pr(1, 'feat/a'), pr(2, 'feat/b')])

    await refreshPullRequests({ '/repo': ['feat/a'] }, true)

    expect(narrow).toHaveBeenCalledWith('/repo', expect.arrayContaining(['feat/a', 'feat/b']), [])
    expect($pullRequestsByBranch.get()[branchPrKey('/repo', 'feat/b')]).toBeDefined()
  })
})
