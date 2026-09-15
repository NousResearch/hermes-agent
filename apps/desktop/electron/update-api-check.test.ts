/**
 * Tests for electron/update-api-check.ts — the API-first passive update check.
 *
 * Why this exists: every desktop client used to `git fetch` twice every 30
 * minutes. GitHub measured tens of millions of fetch/clone requests per day
 * from the install base and asked us to poll via the API instead. These pin
 * the two load-bearing contracts: the cache answers passive checks for a full
 * day but invalidates the moment HEAD moves, and the compare payload maps to
 * an honest behind count (never a fabricated one).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  branchTipApiUrl,
  cacheIsFresh,
  githubRepoSlug,
  listLocalCommits,
  parseCompare,
  resolveBehindLocally,
  UPDATE_CHECK_FAILURE_TTL_MS,
  UPDATE_CHECK_TTL_MS
} from './update-api-check'

const SHA_A = 'a'.repeat(40)
const SHA_B = 'b'.repeat(40)
const HOUR = 60 * 60 * 1000

test('cache serves a passive check for 24h, but not once HEAD or the branch changes', () => {
  const cached = { fetchedAt: 0, currentSha: SHA_A, branch: 'main', status: { behind: 0 } }

  assert.equal(cacheIsFresh(cached, { branch: 'main', currentSha: SHA_A, now: UPDATE_CHECK_TTL_MS - 1 }), true)
  assert.equal(cacheIsFresh(cached, { branch: 'main', currentSha: SHA_A, now: UPDATE_CHECK_TTL_MS }), false)
  // Applying an update moves HEAD: a stale "update available" must never survive it.
  assert.equal(cacheIsFresh(cached, { branch: 'main', currentSha: SHA_B, now: 1 }), false)
  assert.equal(cacheIsFresh(cached, { branch: 'bb/gui', currentSha: SHA_A, now: 1 }), false)

  // Failures retry sooner than successes, but still not on every tick.
  const failed = { ...cached, status: { error: 'fetch-failed' } }
  assert.equal(cacheIsFresh(failed, { branch: 'main', currentSha: SHA_A, now: UPDATE_CHECK_FAILURE_TTL_MS - 1 }), true)
  assert.equal(cacheIsFresh(failed, { branch: 'main', currentSha: SHA_A, now: 2 * HOUR }), false)
})

test('compare payload maps to the behind count and a newest-first commit list; malformed = null', () => {
  const payload = {
    ahead_by: 2,
    status: 'ahead',
    commits: [
      {
        sha: SHA_A,
        commit: { message: 'fix: older\n\nbody', author: { name: 'A' }, committer: { date: '2026-09-10T00:00:00Z' } }
      },
      {
        sha: SHA_B,
        commit: { message: 'feat: newer', author: { name: 'B' }, committer: { date: '2026-09-10T01:00:00Z' } }
      }
    ]
  }

  const parsed = parseCompare(payload)
  assert.equal(parsed?.behind, 2)
  assert.deepEqual(
    parsed?.commits.map(c => [c.sha, c.summary, c.author]),
    [
      [SHA_B, 'feat: newer', 'B'],
      [SHA_A, 'fix: older', 'A']
    ]
  )

  assert.equal(parseCompare({ ahead_by: -1 }), null)
  assert.equal(parseCompare({ status: 'ahead' }), null)
  assert.equal(parseCompare('nope'), null)

  // Forks and SSH forms hit the API for their own repo; non-GitHub origins don't.
  assert.equal(githubRepoSlug('git@github.com:Someone/hermes-agent.git'), 'someone/hermes-agent')
  assert.equal(githubRepoSlug('https://gitlab.example/x/y.git'), null)
  assert.equal(
    branchTipApiUrl('nousresearch/hermes-agent', 'bb/gui'),
    'https://api.github.com/repos/nousresearch/hermes-agent/commits/bb%2Fgui'
  )
})

/**
 * Fake runGit keyed on the subcommand. Records every call so tests can pin
 * that a stale tip short-circuits before any graph walk.
 */
function fakeGit(responses: Record<string, { code: number; stdout?: string }>) {
  const calls: string[][] = []

  return {
    calls,
    runGit: async (args: string[]) => {
      calls.push(args)
      const canned = responses[args[0]] ?? { code: 0 }

      return { code: canned.code, stdout: canned.stdout ?? '', stderr: '' }
    }
  }
}

test('resolveBehindLocally: unreachable tip is unknown, reachable tip is ahead, otherwise the real gap', async () => {
  // A tip missing from the object database (truly stale checkout) stays unknown.
  const stale = fakeGit({ 'cat-file': { code: 1 } })
  assert.equal(await resolveBehindLocally(stale.runGit, '/repo', SHA_A, SHA_B), null)
  assert.deepEqual(stale.calls.map(args => args[0]), ['cat-file'])

  // The remote tip reachable from HEAD is a local commit AHEAD, not an update.
  const ahead = fakeGit({})
  assert.equal(await resolveBehindLocally(ahead.runGit, '/repo', SHA_A, SHA_B), 0)
  assert.deepEqual(ahead.calls.map(args => args[0]), ['cat-file', 'merge-base'])

  // Otherwise the honest local count of HEAD..tip (merge-base must fail first).
  const behind = fakeGit({ 'merge-base': { code: 1 }, 'rev-list': { code: 0, stdout: '3\n' } })
  assert.equal(await resolveBehindLocally(behind.runGit, '/repo', SHA_A, SHA_B), 3)
  assert.deepEqual(behind.calls.map(args => args[0]), ['cat-file', 'merge-base', 'rev-list'])

  // A git failure mid-walk is never silently read as zero.
  const broken = fakeGit({ 'merge-base': { code: 1 }, 'rev-list': { code: 128 } })
  assert.equal(await resolveBehindLocally(broken.runGit, '/repo', SHA_A, SHA_B), null)
})

test('listLocalCommits renders the local gap newest-first in the parseCompare shape', async () => {
  const OLDEST = '1'.repeat(40)
  const NEWEST = '2'.repeat(40)
  const gitLog = fakeGit({
    log: {
      code: 0,
      stdout: [
        [OLDEST, 'Old Hand', '2026-09-10T00:00:00+00:00', 'fix: older'].join('\x1f'),
        [NEWEST, 'New Face', '2026-09-11T00:00:00+00:00', 'feat: newer'].join('\x1f'),
        ''
      ].join('\n')
    }
  })

  const commits = await listLocalCommits(gitLog.runGit, '/repo', SHA_A, SHA_B)
  assert.deepEqual(
    commits.map(c => [c.sha, c.summary, c.author]),
    [
      [NEWEST, 'feat: newer', 'New Face'],
      [OLDEST, 'fix: older', 'Old Hand']
    ]
  )
  assert.equal(commits[0].at, Date.parse('2026-09-11T00:00:00+00:00'))

  // A failed log renders as "no listed commits", never a fabricated list.
  const failed = fakeGit({ log: { code: 128 } })
  assert.deepEqual(await listLocalCommits(failed.runGit, '/repo', SHA_A, SHA_B), [])
})
