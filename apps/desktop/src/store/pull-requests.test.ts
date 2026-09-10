import { beforeEach, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

const { scan, prList } = vi.hoisted(() => ({ scan: vi.fn(), prList: vi.fn() }))
vi.mock('@/hermes', () => ({ scanSessionPullRequests: scan }))
vi.mock('@/lib/desktop-git', () => ({ desktopGit: () => ({ review: { prList } }) }))

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  scan.mockReset()
  prList.mockReset()
})

it('recovers the PR actually opened even when a session already records another branch, and retries changed transcripts', async () => {
  const store = await import('./pull-requests')

  const session = {
    id: 'worktree',
    git_repo_root: '/repo',
    git_branch: 'integration',
    message_count: 1,
    last_active: 1
  } as SessionInfo

  scan.mockResolvedValueOnce({ pull_requests: {}, scanned: [session.id] })
  await store.recoverSessionPullRequests([session])
  expect(scan).toHaveBeenCalledOnce()
  await store.recoverSessionPullRequests([session])
  expect(scan).toHaveBeenCalledOnce()
  const url = 'https://github.com/example/project/pull/42'
  scan.mockResolvedValueOnce({ pull_requests: { [session.id]: { number: 42, url } }, scanned: [session.id] })
  await store.recoverSessionPullRequests([{ ...session, message_count: 3, last_active: 2 }])
  expect(scan).toHaveBeenCalledTimes(2)
  expect(store.sessionPrKey(session)).toBe(store.numberPrKey('https://github.com/example/project', 42))
  const unbound = { ...session, id: 'outside-git', git_repo_root: null, git_branch: null }
  scan.mockResolvedValueOnce({ pull_requests: { [unbound.id]: { number: 42, url } }, scanned: [unbound.id] })
  await store.recoverSessionPullRequests([unbound])
  expect(store.sessionPrKey(unbound)).toBe(store.sessionPrKey(session))
  prList.mockResolvedValue({ prs: [{ number: 42, url, branch: 'fix', title: 'Fix', state: 'open', draft: false }] })
  await store.refreshPullRequests({ 'https://github.com/example/project': ['#42'] })
  expect(prList).toHaveBeenCalledWith('', [], [], [url])
  expect(store.$pullRequestsByBranch.get()[store.sessionPrKey(unbound)!]?.url).toBe(url)
})

it.each(['before', 'during'])(
  'preserves a desktop-created association stamped %s an older transcript scan',
  async timing => {
    const store = await import('./pull-requests')
    const session = { id: 'replacement', message_count: 3, last_active: 2 } as SessionInfo
    let finishScan!: (value: unknown) => void
    scan.mockImplementation(
      () =>
        new Promise(resolve => {
          finishScan = resolve
        })
    )

    if (timing === 'before') {
      store.stampSessionPrBranch(session.id, '/repo', 'replacement')
    }

    const recovery = store.recoverSessionPullRequests([session])

    if (timing === 'during') {
      store.stampSessionPrBranch(session.id, '/repo', 'replacement')
    }

    finishScan({
      pull_requests: { [session.id]: { number: 42, url: 'https://github.com/example/project/pull/42' } },
      scanned: [session.id]
    })
    await recovery
    expect(store.sessionPrKey(session)).toBe(store.branchPrKey('/repo', 'replacement'))
  }
)

it('fetches only current lookups while preserving unrelated cache entries and their own freshness', async () => {
  const store = await import('./pull-requests')
  const now = vi.spyOn(Date, 'now').mockReturnValue(100_000)
  const branches = Array.from({ length: 300 }, (_, i) => `old-${i}`)

  const found = {
    number: 42,
    url: 'https://github.com/example/project/pull/42',
    branch: 'new-visible',
    title: 'Fix',
    state: 'open',
    draft: false
  }

  const old = { ...found, number: 41, url: 'https://github.com/example/project/pull/41', branch: 'old-0' }
  prList.mockImplementation(async (_root, requested: string[]) => ({
    ghReady: true,
    prs: [old, found].filter(pr => requested.slice(0, 300).includes(pr.branch))
  }))

  try {
    await store.refreshPullRequests({ '/repo': branches })
    now.mockReturnValue(130_001)
    await store.refreshPullRequests({ '/repo': ['new-visible'] })
    const oldKey = store.branchPrKey('/repo', old.branch)
    const newKey = store.branchPrKey('/repo', found.branch)
    expect(store.$pullRequestsByBranch.get()[newKey]).toEqual(found)
    expect(prList.mock.calls[1]).toEqual(['/repo', ['new-visible'], []])
    expect(store.$pullRequestsByBranch.get()[oldKey]).toEqual(old)

    now.mockReturnValue(160_001)
    await store.refreshPullRequests({ '/repo': ['old-0'] })
    expect(prList).toHaveBeenCalledTimes(3)
    expect(prList.mock.calls[2]).toEqual(['/repo', ['old-0'], []])
    await store.refreshPullRequests({ '/repo': ['new-visible'] })
    expect(prList).toHaveBeenCalledTimes(3)

    prList.mockResolvedValue({ ghReady: true, prs: [] })
    await store.refreshPullRequests({ '/repo': ['old-0'] }, true)
    expect(store.$pullRequestsByBranch.get()[oldKey]).toBeUndefined()
    expect(store.$pullRequestsByBranch.get()[newKey]).toEqual(found)
  } finally {
    now.mockRestore()
  }
})

it.each(['/repo', 'https://github.com/example/project'])(
  'hydrates every requested lookup beyond the API cap for %s',
  async root => {
    const store = await import('./pull-requests')

    const prs = Array.from({ length: 301 }, (_, i) => ({
      number: i + 1,
      url: `https://github.com/example/project/pull/${i + 1}`,
      branch: `branch-${i}`,
      title: 'Fix',
      state: 'open',
      draft: false
    }))

    const lookups = prs.map((pr, i) => (root === '/repo' && i % 2 === 0 ? pr.branch : `#${pr.number}`))
    prList.mockImplementation(async (_root, branches: string[], numbers: number[] = [], urls: string[] = []) => ({
      ghReady: true,
      prs: prs
        .filter(pr => branches.includes(pr.branch) || numbers.includes(pr.number) || urls.includes(pr.url))
        .slice(0, 300)
    }))
    await store.refreshPullRequests({ [root]: [...lookups, lookups[0]] })
    expect(prList).toHaveBeenCalledTimes(2)

    for (const [, branches, numbers = [], urls = []] of prList.mock.calls) {
      expect(branches.length + numbers.length + urls.length).toBeLessThanOrEqual(300)
    }

    for (const [i, lookup] of lookups.entries()) {
      expect(store.$pullRequestsByBranch.get()[store.branchPrKey(root, lookup)]).toEqual(prs[i])
    }

    await store.refreshPullRequests({ [root]: lookups })
    expect(prList).toHaveBeenCalledTimes(2)
  }
)

it('fetches newly requested lookups without waiting for the previous repo query to become stale', async () => {
  const store = await import('./pull-requests')
  let finishList!: (value: unknown) => void
  prList.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finishList = resolve
      })
  )
  prList.mockResolvedValue({ prs: [] })
  const initial = store.refreshPullRequests({ '/repo': ['integration'] })
  const expanded = store.refreshPullRequests({ '/repo': ['integration', '#42'] })
  const duplicate = store.refreshPullRequests({ '/repo': ['integration', '#42'] })
  expect(prList).toHaveBeenCalledOnce()
  finishList({ prs: [] })
  await Promise.all([initial, expanded, duplicate])
  expect(prList).toHaveBeenCalledTimes(2)
  expect(prList.mock.calls[1]).toEqual(['/repo', ['integration'], [42]])
  const key = store.numberPrKey('/repo', 42)

  const known = {
    number: 42,
    url: 'https://github.com/example/project/pull/42',
    branch: 'fix',
    title: 'Fix',
    state: 'open' as const,
    draft: false
  }

  store.$pullRequestsByBranch.set({ [key]: known })
  prList.mockResolvedValue({ ghReady: false, prs: [] })
  await store.refreshPullRequests({ '/repo': ['#42'] }, true)
  expect(store.$pullRequestsByBranch.get()[key]).toEqual(known)
  prList.mockRejectedValueOnce(new Error('network unavailable'))
  await store.refreshPullRequests({ '/repo': ['#42'] }, true)
  expect(store.$pullRequestsByBranch.get()[key]).toEqual(known)
  prList.mockResolvedValue({ ghReady: true, prs: [] })
  await store.refreshPullRequests({ '/repo': ['#42'] }, true)
  expect(store.$pullRequestsByBranch.get()[key]).toBeUndefined()
})
