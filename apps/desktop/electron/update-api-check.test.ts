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
  parseGitHubToken,
  resolveGitHubToken,
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

test('a .env token is found verbatim, and an absent or unrelated file yields no header', () => {
  assert.equal(parseGitHubToken('ANTHROPIC_API_KEY=x\nGITHUB_TOKEN=ghp_abc123\n'), 'ghp_abc123')
  assert.equal(parseGitHubToken('GH_TOKEN="gho_quoted"\n'), 'gho_quoted')
  assert.equal(parseGitHubToken('  GITHUB_TOKEN = ghp_spaced  \n'), 'ghp_spaced')
  // No token: the caller must send no Authorization header, not an empty Bearer.
  assert.equal(parseGitHubToken('OPENAI_API_KEY=x\n# GITHUB_TOKEN=commented\n'), '')
  assert.equal(parseGitHubToken(''), '')
  // A key that merely mentions the name is not a token.
  assert.equal(parseGitHubToken('GITHUB_TOKEN_FILE=/tmp/x\n'), '')
})

test('the credential chain stops at the first source that answers, and never asks a later one', async () => {
  let ghCalls = 0

  const gh = async () => {
    ghCalls += 1

    return 'gho_from_gh\n'
  }

  // Env wins: neither the file nor the CLI is consulted.
  assert.equal(await resolveGitHubToken('ghp_env', () => 'GITHUB_TOKEN=ghp_file\n', gh), 'ghp_env')
  assert.equal(ghCalls, 0)

  // Then the file, before the CLI.
  assert.equal(await resolveGitHubToken('', () => 'GITHUB_TOKEN=ghp_file\n', gh), 'ghp_file')
  assert.equal(ghCalls, 0)

  // Then the CLI, trimmed.
  assert.equal(await resolveGitHubToken('', () => '', gh), 'gho_from_gh')
  assert.equal(ghCalls, 1)

  // Nobody has a credential: anonymous, i.e. no header — and a CLI that fails
  // (missing gh, not logged in, timeout) must not throw into the update check.
  assert.equal(await resolveGitHubToken('', () => '', async () => ''), '')
  assert.equal(
    await resolveGitHubToken(
      '',
      () => '',
      async () => {
        throw new Error('spawn gh ENOENT')
      }
    ),
    ''
  )
})
