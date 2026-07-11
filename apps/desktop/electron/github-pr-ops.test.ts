import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildPullRequestSearchArgs,
  getGithubAuthStatus,
  getGithubPullRequestDetail,
  listGithubPullRequests,
  normalizeCheckSummary,
  normalizePullRequestDetail,
  normalizePullRequestSummary,
  SUMMARY_FIELDS,
  validatePullRequestFilter,
  validatePullRequestRef
} from './github-pr-ops'

test('builds fixed created and review-requested searches', () => {
  assert.deepEqual(buildPullRequestSearchArgs({ kind: 'created', state: 'open' }).slice(0, 5), [
    'search',
    'prs',
    '--author',
    '@me',
    '--state'
  ])
  assert.ok(buildPullRequestSearchArgs({ kind: 'created', state: 'closed' }).includes('closed'))
  assert.ok(buildPullRequestSearchArgs({ kind: 'review-requested', state: 'open' }).includes('--review-requested'))
  assert.equal(buildPullRequestSearchArgs({ kind: 'created', state: 'open', limit: 999 }).at(-3), '100')
  assert.ok(buildPullRequestSearchArgs({ kind: 'created', state: 'open' }).at(-1)?.includes(SUMMARY_FIELDS))
})

test('validates filters and structured references', () => {
  assert.throws(() => validatePullRequestFilter({ kind: 'review-requested', state: 'closed' }))
  assert.deepEqual(validatePullRequestRef({ repository: 'NousResearch/hermes-agent', number: 1 }), {
    repository: 'NousResearch/hermes-agent',
    number: 1
  })

  for (const repository of ['owner', 'owner/repo/extra', 'owner/repo;echo']) {
    assert.throws(() => validatePullRequestRef({ repository, number: 1 }))
  }
  assert.throws(() => validatePullRequestRef({ repository: 'owner/repo', number: 0 }))
})

test('normalizes summaries defensively', () => {
  const item = normalizePullRequestSummary({
    repository: { nameWithOwner: 'o/r' },
    number: 4,
    title: 'PR',
    url: 'https://github.com/o/r/pull/4',
    state: 'open',
    isDraft: true
  })
  assert.equal(item?.state, 'OPEN')
  assert.equal(item?.isDraft, true)
  assert.equal(item?.author, null)
  assert.deepEqual(item?.labels, [])
  assert.equal(normalizePullRequestSummary({ repository: {}, number: 1, title: 'x', url: 'x' }), null)
})

test('normalizes merged details and check rollups', () => {
  const checks = normalizeCheckSummary([
    { status: 'COMPLETED', conclusion: 'SUCCESS' },
    { status: 'COMPLETED', conclusion: 'FAILURE' },
    { status: 'IN_PROGRESS' }
  ])
  assert.deepEqual(checks, { total: 3, pending: 1, passed: 1, failed: 1, skipped: 0 })
  const detail = normalizePullRequestDetail(
    { number: 2, title: 'x', url: 'u', state: 'closed', mergedAt: 'now', statusCheckRollup: [] },
    'o/r'
  )
  assert.equal(detail?.state, 'MERGED')
})

test('returns structured missing, auth, malformed, and CLI failure states', async () => {
  assert.equal(await getGithubAuthStatus('gh', async () => ({ ok: false, kind: 'missing' })), 'gh-missing')
  assert.equal(
    (
      await listGithubPullRequests({ kind: 'created', state: 'open' }, 'gh', async args =>
        args[0] === 'auth' ? { ok: true, kind: 'success' } : { ok: true, kind: 'success', stdout: '{' }
      )
    ).authState,
    'error'
  )
  assert.equal(
    (
      await listGithubPullRequests({ kind: 'created', state: 'open' }, 'gh', async args =>
        args[0] === 'auth' ? { ok: true, kind: 'success' } : { ok: false, kind: 'timeout' }
      )
    ).error,
    'GitHub CLI request timed out'
  )
})

test('detail uses a fixed argument array', async () => {
  let captured

  const detail = await getGithubPullRequestDetail({ repository: 'o/r', number: 7 }, 'gh', async args => {
    captured = args

    return { ok: true, kind: 'success', stdout: JSON.stringify({ number: 7, title: 'x', url: 'u', state: 'open' }) }
  })

  assert.deepEqual(captured?.slice(0, 6), ['pr', 'view', '7', '--repo', 'o/r', '--json'])
  assert.equal(detail.number, 7)
})
