import assert from 'node:assert/strict'

import { test } from 'vitest'

import { excerptSpawnLog } from './spawn-log-excerpt'

test('excerptSpawnLog strips ANSI, secrets, and bounds the tail', () => {
  const raw = [
    '\u001B[31mTraceback (most recent call last):\u001B[0m',
    'HERMES_DASHBOARD_SESSION_TOKEN=supersecret',
    'Authorization: Bearer tok_9999',
    'IdentityFile=/Users/me/.ssh/id_ed25519',
    'cat ~/.hermes/desktop-ssh/0123456789abcdef0123456789abcdef/deadbeef.log',
    'PermissionError: denied',
    'OOM killed'
  ].join('\n')

  const excerpt = excerptSpawnLog(raw)
  assert.match(excerpt, /PermissionError/)
  assert.match(excerpt, /OOM killed/)
  assert.doesNotMatch(excerpt, /supersecret|tok_9999|id_ed25519|desktop-ssh|cat /)
  assert.doesNotMatch(excerpt, /\u001B/)
})

test('excerptSpawnLog fail-opens on empty or whitespace-only input', () => {
  assert.equal(excerptSpawnLog(''), '')
  assert.equal(excerptSpawnLog('   \n\t\n'), '')
  assert.equal(excerptSpawnLog(null), '')
})
