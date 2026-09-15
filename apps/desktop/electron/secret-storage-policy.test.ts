import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  classifyStoredSecret,
  readSecretStoragePolicy,
  SECRET_STORAGE_POLICY_VERSION,
  type SecretStoragePolicyIo,
  writeSecretStoragePolicy
} from './secret-storage-policy'

function fakeIo(initial: string | null = null): SecretStoragePolicyIo & { fileText: () => string | null } {
  let text = initial

  return {
    readText: () => {
      if (text === null) {
        throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
      }

      return text
    },
    writeText: (next: string) => {
      text = next
    },
    fileText: () => text
  }
}

const SECURE_DEFAULT = { on: true, migrated: false }

test('missing or malformed policy fails closed to encryption ON', () => {
  for (const input of [null, 'not-json', '[]', '"on"', 'null', '123']) {
    assert.deepEqual(readSecretStoragePolicy(fakeIo(input)), SECURE_DEFAULT)
  }
})

test('legacy two-field plaintext records do not masquerade as explicit consent', () => {
  assert.deepEqual(readSecretStoragePolicy(fakeIo('{"on":false,"migrated":true}')), SECURE_DEFAULT)
  assert.deepEqual(readSecretStoragePolicy(fakeIo('{"on":true,"migrated":false}')), SECURE_DEFAULT)
})

test('only the exact current schema is authoritative', () => {
  const invalid = [
    '{"version":1,"on":false,"migrated":true}',
    '{"version":2,"on":false}',
    '{"version":2,"on":0,"migrated":true}',
    '{"version":2,"on":false,"migrated":true,"unknown":false}',
    '{"version":2,"on":true,"on":false,"migrated":true}',
    '{"version":2,"on":false,"migrated":true} trailing'
  ]

  for (const input of invalid) {
    assert.deepEqual(readSecretStoragePolicy(fakeIo(input)), SECURE_DEFAULT)
  }
})

test('current-schema explicit opt-out and opt-in are both honored', () => {
  assert.deepEqual(
    readSecretStoragePolicy(fakeIo('{"version":2,"on":false,"migrated":true}')),
    { on: false, migrated: true }
  )
  assert.deepEqual(
    readSecretStoragePolicy(fakeIo('{"version":2,"on":true,"migrated":false}')),
    { on: true, migrated: false }
  )
})

test('writes the current schema and round-trips both policy states', () => {
  const io = fakeIo()

  writeSecretStoragePolicy({ on: false, migrated: true }, io)
  assert.deepEqual(JSON.parse(io.fileText() || ''), {
    version: SECRET_STORAGE_POLICY_VERSION,
    on: false,
    migrated: true
  })
  assert.deepEqual(readSecretStoragePolicy(io), { on: false, migrated: true })

  writeSecretStoragePolicy({ on: true, migrated: false }, io)
  assert.deepEqual(readSecretStoragePolicy(io), { on: true, migrated: false })
})

const SAFE_BLOB = { encoding: 'safeStorage', value: 'AAAA' }
const PLAIN_BLOB = { encoding: 'plain', value: 'tok' }

test('non-safeStorage blobs are always kept', () => {
  for (const policy of [
    { on: false, migrated: false },
    { on: false, migrated: true },
    { on: true, migrated: false }
  ]) {
    assert.equal(classifyStoredSecret(PLAIN_BLOB, policy), 'keep')
    assert.equal(classifyStoredSecret(null, policy), 'keep')
    assert.equal(classifyStoredSecret(undefined, policy), 'keep')
    assert.equal(classifyStoredSecret({}, policy), 'keep')
  }
})

test('safeStorage blobs stay available while encryption is on', () => {
  assert.equal(classifyStoredSecret(SAFE_BLOB, { on: true, migrated: false }), 'keep')
})

test('explicit opt-out migrates once and then drops undecryptable blobs without another keychain touch', () => {
  assert.equal(classifyStoredSecret(SAFE_BLOB, { on: false, migrated: false }), 'migrate')
  assert.equal(classifyStoredSecret(SAFE_BLOB, { on: false, migrated: true }), 'drop')
})
