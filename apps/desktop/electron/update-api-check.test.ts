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
  resolveGitHubApiToken,
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

test('token resolution: PAT env first, gh CLI second, anonymous on any failure', async () => {
  // A PAT in the environment wins over gh, and is trimmed rather than sent raw.
  assert.equal(
    await resolveGitHubApiToken({ env: { GITHUB_TOKEN: ' ghp_pat ' }, ghAuthToken: async () => 'ghp_cli' }),
    'ghp_pat'
  )
  // GH_TOKEN is the documented alias.
  assert.equal(await resolveGitHubApiToken({ env: { GH_TOKEN: 'ghp_alias' } }), 'ghp_alias')

  // Whitespace-only vars fall through to gh instead of sending a junk header.
  assert.equal(
    await resolveGitHubApiToken({ env: { GITHUB_TOKEN: '   ' }, ghAuthToken: async () => ' ghp_cli ' }),
    'ghp_cli'
  )

  // No credentials anywhere (gh missing, logged out, or hung) resolves null:
  // the request goes out anonymous — the pre-fix behavior, never an error the
  // update check has to handle (#108804).
  assert.equal(await resolveGitHubApiToken({ env: {}, ghAuthToken: async () => null }), null)
  assert.equal(await resolveGitHubApiToken({ env: {} }), null)
  assert.equal(
    await resolveGitHubApiToken({ env: {}, ghAuthToken: async () => { throw new Error('ENOENT') } }),
    null
  )
})
