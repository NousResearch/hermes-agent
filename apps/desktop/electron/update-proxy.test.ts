import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildUpdateCheckEnv,
  hasProxyEnv,
  isGitNetworkError,
  parseMacSystemProxy,
  withoutProxyEnv
} from './update-proxy'

test('parseMacSystemProxy reads enabled HTTP and HTTPS proxies', () => {
  const env = parseMacSystemProxy(`
    HTTPEnable : 1
    HTTPPort : 7897
    HTTPProxy : 127.0.0.1
    HTTPSEnable : 1
    HTTPSPort : 7897
    HTTPSProxy : 127.0.0.1
  `)

  assert.equal(env.http_proxy, 'http://127.0.0.1:7897')
  assert.equal(env.https_proxy, 'http://127.0.0.1:7897')
  assert.equal(env.all_proxy, 'http://127.0.0.1:7897')
})

test('parseMacSystemProxy ignores disabled or incomplete entries', () => {
  assert.deepEqual(
    parseMacSystemProxy('HTTPEnable : 0\nHTTPProxy : 127.0.0.1\nHTTPPort : 7897'),
    {}
  )
})

test('buildUpdateCheckEnv preserves inherited proxy values', () => {
  const env = buildUpdateCheckEnv({ https_proxy: 'http://inherited:8080' }, 'linux')

  assert.equal(env.https_proxy, 'http://inherited:8080')
  assert.equal(env.GIT_TERMINAL_PROMPT, '0')
})

test('withoutProxyEnv removes all common proxy spellings', () => {
  assert.equal(hasProxyEnv(withoutProxyEnv({ http_proxy: 'x', HTTPS_PROXY: 'y' })), false)
})

test('isGitNetworkError identifies transport failures but not branch errors', () => {
  assert.equal(isGitNetworkError('fatal: unable to access https://github.com: Failed to connect'), true)
  assert.equal(isGitNetworkError('fatal: Remote branch missing not found'), false)
})
