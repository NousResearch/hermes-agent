import assert from 'node:assert/strict'

import { test } from 'vitest'

import { githubApiHeaders, githubTokenFromEnv, resolveGithubToken } from './github-api-auth'

// The exact headers the anonymous update check sends today; the change must
// leave this shape byte-identical when no token is configured.
const UPDATE_CHECK_HEADERS = {
  Accept: 'application/vnd.github.sha',
  'User-Agent': 'hermes-desktop-update-check'
}

test('GITHUB_TOKEN wins over GH_TOKEN', () => {
  assert.equal(githubTokenFromEnv({ GITHUB_TOKEN: 'pat-a', GH_TOKEN: 'pat-b' }), 'pat-a')
})

test('a blank env token falls through to the next var, and the value is trimmed', () => {
  assert.equal(githubTokenFromEnv({ GITHUB_TOKEN: '   ', GH_TOKEN: '  pat-b  ' }), 'pat-b')
  assert.equal(githubTokenFromEnv({ GITHUB_TOKEN: '', GH_TOKEN: '' }), null)
  assert.equal(githubTokenFromEnv({}), null)
})

test('the anonymous path sends no Authorization key at all', () => {
  const headers = githubApiHeaders(UPDATE_CHECK_HEADERS, null)

  assert.deepEqual(headers, UPDATE_CHECK_HEADERS)
  assert.equal('Authorization' in headers, false)
})

test('a token adds the `token`-scheme Authorization header without disturbing the base headers', () => {
  const headers = githubApiHeaders(UPDATE_CHECK_HEADERS, 'pat-a')

  assert.equal(headers.Authorization, 'token pat-a')
  assert.equal(headers['User-Agent'], 'hermes-desktop-update-check')
  // The caller's base object is a constant; mutating it would leak the token
  // into every later request that builds from it.
  assert.equal('Authorization' in UPDATE_CHECK_HEADERS, false)
})

test('the gh CLI is consulted only when the env rung is empty', async () => {
  let ghCalls = 0

  const readGhCliToken = async () => {
    ghCalls += 1

    return 'gh-token'
  }

  assert.deepEqual(await resolveGithubToken({ env: { GH_TOKEN: 'pat-a' }, readGhCliToken }), {
    token: 'pat-a',
    source: 'env'
  })
  assert.equal(ghCalls, 0)

  assert.deepEqual(await resolveGithubToken({ env: {}, readGhCliToken }), { token: 'gh-token', source: 'gh-cli' })
  assert.equal(ghCalls, 1)
})

test('a missing, blank or failing gh degrades to the anonymous path instead of throwing', async () => {
  const missing = async () => null
  const blank = async () => '   '

  const failing = async () => {
    throw new Error('spawn gh ENOENT')
  }

  assert.deepEqual(await resolveGithubToken({ readGhCliToken: missing }), { token: null, source: 'none' })
  assert.deepEqual(await resolveGithubToken({ readGhCliToken: blank }), { token: null, source: 'none' })
  assert.deepEqual(await resolveGithubToken({ readGhCliToken: failing }), { token: null, source: 'none' })
})
