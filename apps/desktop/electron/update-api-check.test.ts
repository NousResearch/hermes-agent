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
  parseCompare,
  UPDATE_CHECK_FAILURE_TTL_MS,
  UPDATE_CHECK_TTL_MS
} from './update-api-check'

const SHA_A = 'a'.repeat(40)
const SHA_B = 'b'.repeat(40)
const HOUR = 60 * 60 * 1000
const REPOSITORY = 'nousresearch/hermes-agent'

test('cache serves a passive check for 24h, but not once HEAD or the branch changes', () => {
  const cached = { fetchedAt: 0, currentSha: SHA_A, branch: 'main', status: { behind: 0, repository: REPOSITORY } }
  const current = { branch: 'main', currentSha: SHA_A, repository: REPOSITORY }

  assert.equal(cacheIsFresh(cached, { ...current, now: UPDATE_CHECK_TTL_MS - 1 }), true)
  assert.equal(cacheIsFresh(cached, { ...current, now: UPDATE_CHECK_TTL_MS }), false)
  // Applying an update moves HEAD: a stale "update available" must never survive it.
  assert.equal(cacheIsFresh(cached, { ...current, currentSha: SHA_B, now: 1 }), false)
  assert.equal(cacheIsFresh(cached, { ...current, branch: 'bb/gui', now: 1 }), false)

  // Failures retry sooner than successes, but still not on every tick.
  const failed = { ...cached, status: { ...cached.status, error: 'fetch-failed' } }
  assert.equal(cacheIsFresh(failed, { ...current, now: UPDATE_CHECK_FAILURE_TTL_MS - 1 }), true)
  assert.equal(cacheIsFresh(failed, { ...current, now: 2 * HOUR }), false)
})

test('cache belongs to its source repository, never a different fork or an unidentified legacy source', () => {
  const cached = { fetchedAt: 0, currentSha: SHA_A, branch: 'main', status: { behind: 0, repository: REPOSITORY } }

  const current = {
    branch: 'main',
    currentSha: SHA_A,
    now: 1,
    repository: githubRepoSlug('git@github.com:NousResearch/hermes-agent.git')
  }

  // HTTPS/SSH and casing differences still identify the same repository.
  assert.equal(cacheIsFresh(cached, current), true)
  assert.equal(
    cacheIsFresh(cached, { ...current, repository: githubRepoSlug('https://github.com/someone/hermes-agent.git') }),
    false
  )

  const legacy = { ...cached, status: { behind: 0 } }
  assert.equal(cacheIsFresh(legacy, current), false)
  assert.equal(cacheIsFresh(cached, { ...current, repository: null }), false)
  assert.equal(cacheIsFresh(legacy, { ...current, repository: null }), false)
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
