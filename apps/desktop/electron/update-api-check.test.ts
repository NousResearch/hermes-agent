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
  describeUpdateCheckFailure,
  githubRepoSlug,
  parseCompare,
  rateLimitFromHeaders,
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

// #112615: behind a shared exit IP the anonymous 60/hour budget is spent by
// neighbours, so "try again in an hour" is advice the user cannot act on. A
// genuine rate limit (x-ratelimit-remaining: 0) must name the shared-address
// cause, the real reset time and the GITHUB_TOKEN remedy; any other 403 must
// not be reported as a rate limit.
test('a rate-limited 403 names the shared-IP cause, the reset time and GITHUB_TOKEN; a plain 403 is not a rate limit', () => {
  const now = 1_700_000_000_000

  const limited = {
    statusCode: 403,
    ...rateLimitFromHeaders({ 'x-ratelimit-remaining': '0', 'x-ratelimit-reset': String(now / 1000 + 25 * 60) }),
    authenticated: false
  }

  const message = describeUpdateCheckFailure(limited, now)

  assert.match(message, /60 per hour per network address/)
  assert.match(message, /in about 25 minutes/)
  assert.match(message, /GITHUB_TOKEN/)

  assert.match(describeUpdateCheckFailure({ ...limited, authenticated: true }, now), /for your GITHUB_TOKEN/)

  // Missing or non-zero rate-limit headers: an ordinary 403, reported as such.
  assert.equal(describeUpdateCheckFailure({ statusCode: 403 }), 'api.github.com answered HTTP 403.')
  assert.equal(
    describeUpdateCheckFailure({ statusCode: 403, ...rateLimitFromHeaders({ 'x-ratelimit-remaining': '57' }) }),
    'api.github.com answered HTTP 403.'
  )
})
