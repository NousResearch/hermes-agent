import assert from 'node:assert/strict'

import { test } from 'vitest'

import { joinBackendApiUrl } from './backend-api-url'

const LOCAL = 'http://127.0.0.1:7890'
const REMOTE = 'https://gateway.example.com'

test('joinBackendApiUrl keeps local API paths on the loopback origin', () => {
  assert.equal(joinBackendApiUrl(LOCAL, '/api/status'), `${LOCAL}/api/status`)
  assert.equal(
    joinBackendApiUrl(LOCAL, '/api/sessions?limit=50&offset=0'),
    `${LOCAL}/api/sessions?limit=50&offset=0`
  )
})

test('joinBackendApiUrl keeps remote gateway paths on the configured origin', () => {
  assert.equal(joinBackendApiUrl(REMOTE, '/api/sessions'), `${REMOTE}/api/sessions`)
  assert.equal(
    joinBackendApiUrl(`${REMOTE}/`, '/api/model/info?profile=iris'),
    `${REMOTE}/api/model/info?profile=iris`
  )
})

test('joinBackendApiUrl allows path segments that look like userinfo but stay on-origin', () => {
  assert.equal(joinBackendApiUrl(LOCAL, '/@literal'), `${LOCAL}/@literal`)
})

test('joinBackendApiUrl rejects @-authority retargeting that would leak credentials', () => {
  assert.throws(() => joinBackendApiUrl(LOCAL, '@attacker.example/'), /relative path/)
  assert.throws(() => joinBackendApiUrl(REMOTE, '@attacker.example/steal'), /relative path/)
})

test('joinBackendApiUrl rejects protocol-relative and empty paths', () => {
  assert.throws(() => joinBackendApiUrl(LOCAL, '//attacker.example/'), /relative path/)
  assert.throws(() => joinBackendApiUrl(LOCAL, ''), /non-empty/)
  assert.throws(() => joinBackendApiUrl(LOCAL, null), /non-empty/)
  assert.throws(() => joinBackendApiUrl(LOCAL, undefined), /non-empty/)
})

test('joinBackendApiUrl rejects absolute URLs smuggled as the path', () => {
  assert.throws(() => joinBackendApiUrl(LOCAL, 'http://attacker.example/'), /relative path/)
  assert.throws(() => joinBackendApiUrl(LOCAL, 'https://attacker.example/'), /relative path/)
})
