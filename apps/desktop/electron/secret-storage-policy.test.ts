/**
 * Tests for electron/secret-storage-policy.ts — the "is OS-keychain
 * encryption enabled at all?" decision seam.
 *
 * The behavior this file pins: keychain-backed encryption is OPT-IN
 * (default OFF), and once the one-shot legacy migration has run, a
 * safeStorage blob under an opted-out policy reads as 'drop' — i.e. the
 * caller must treat it as absent WITHOUT touching safeStorage, so a broken
 * macOS login keychain can never raise its password dialog on launch.
 *
 * (Wired into the vitest `electron` project via electron/**\/*.test.ts.)
 */

import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { encryptDesktopSecret, writeSecretFileAtomic } from './hardening'
import { roomSetupStore } from './room-setup-store'
import type { SetupRecord } from './room-setup-store'
import {
  classifyStoredSecret,
  readSecretStoragePolicy,
  requireRoomSetupEncryption,
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

// ── defaults ────────────────────────────────────────────────────────────────

test('missing policy file defaults to encryption OFF, not migrated', () => {
  const policy = readSecretStoragePolicy(fakeIo())

  assert.deepEqual(policy, { on: false, migrated: false })
})

test('corrupt or non-object policy file reads as the default', () => {
  for (const bad of ['not-json', '[]', '"on"', 'null', '123']) {
    assert.deepEqual(readSecretStoragePolicy(fakeIo(bad)), { on: false, migrated: false })
  }
})

test('truthy-but-not-true values do NOT enable encryption', () => {
  // Strict === true coercion: a hand-edited "on": 1 or "yes" must not turn
  // keychain prompts back on.
  for (const bad of ['{"on":1}', '{"on":"yes"}', '{"on":"true"}']) {
    assert.equal(readSecretStoragePolicy(fakeIo(bad)).on, false)
  }
})

test('round trip preserves both fields', () => {
  const io = fakeIo()

  writeSecretStoragePolicy({ on: true, migrated: true }, io)
  assert.deepEqual(readSecretStoragePolicy(io), { on: true, migrated: true })

  writeSecretStoragePolicy({ on: false, migrated: true }, io)
  assert.deepEqual(readSecretStoragePolicy(io), { on: false, migrated: true })
})

// ── classification ──────────────────────────────────────────────────────────

const SAFE_BLOB = { encoding: 'safeStorage', value: 'AAAA' }
const PLAIN_BLOB = { encoding: 'plain', value: 'tok' }

test('non-safeStorage blobs are always keep, under every policy', () => {
  for (const policy of [
    { on: false, migrated: false },
    { on: false, migrated: true },
    { on: true, migrated: true }
  ]) {
    assert.equal(classifyStoredSecret(PLAIN_BLOB, policy), 'keep')
    assert.equal(classifyStoredSecret(null, policy), 'keep')
    assert.equal(classifyStoredSecret(undefined, policy), 'keep')
    assert.equal(classifyStoredSecret({} as any, policy), 'keep')
  }
})

test('safeStorage blob with encryption ON is keep', () => {
  assert.equal(classifyStoredSecret(SAFE_BLOB, { on: true, migrated: true }), 'keep')
})

test('safeStorage blob, encryption OFF, pre-migration is migrate', () => {
  assert.equal(classifyStoredSecret(SAFE_BLOB, { on: false, migrated: false }), 'migrate')
})

test('safeStorage blob, encryption OFF, post-migration is drop — never touch the keychain again', () => {
  assert.equal(classifyStoredSecret(SAFE_BLOB, { on: false, migrated: true }), 'drop')
})


test('room puts and ON rotation reject basic_text even when encryption reports available; OFF never probes', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'room-storage-policy-'))
  const policy = { on: false, migrated: true }
  let backend = 'basic_text', backendReads = 0, encrypted = 0
  // Storage capability injection, not pretending this test runs on another OS.
  const api = { isEncryptionAvailable: () => true, encryptString: (value: string) => {
    encrypted++; return Buffer.from(value)
  } }
  const encode = (value: string) => {
    requireRoomSetupEncryption(policy, () => {backendReads++; return backend})
    return policy.on ? encryptDesktopSecret(value, api) : { encoding: 'plain', value }
  }
  const decode = (value: string) => {
    const secret = JSON.parse(value)
    return secret.encoding === 'plain' ? secret.value : Buffer.from(secret.value, 'base64').toString()
  }
  const store = roomSetupStore({ directory, encrypt: value => JSON.stringify(encode(value)), decrypt: decode })
  const record: SetupRecord = { id: crypto.randomUUID(), setupId: crypto.randomUUID(), kind: 'peer',
    roomId: 'room', installationId: 'original', route: { connectionId: 'peer', profile: 'default' }, grant: 'pending-grant' }
  const destination = path.join(directory, record.id + '.json')
  const rotate = async () => {
    const original = decode(await fs.readFile(destination, 'utf8'))
    const next = encode(original)
    writeSecretFileAtomic(destination, JSON.stringify(next), { encoding: 'utf8', durable: {
      verify: bytes => assert.equal(decode(bytes.toString()), original)
    } })
  }
  try {
    await store.put(record)
    assert.equal(backendReads, 0)
    assert.equal(encrypted, 0)
    const pending = await fs.readFile(destination)
    policy.on = true
    await assert.rejects(store.put({ ...record, grant: 'fresh-grant' }), /secure_storage_required/)
    await assert.rejects(rotate(), /secure_storage_required/)
    assert.equal(encrypted, 0)
    assert.deepEqual(await fs.readFile(destination), pending)
    assert.deepEqual(await store.get(record.id), record)
    // The generic helper alone accepts this misleading capability; that is why
    // both room write entrances need their shared selected-backend guard.
    assert.equal(encryptDesktopSecret('probe', api).encoding, 'safeStorage')
    backend = 'gnome_libsecret'
    await rotate()
    await store.put({ ...record, grant: 'fresh-grant' })
    assert.equal((await store.get(record.id)).grant, 'fresh-grant')
  } finally {await fs.rm(directory, { recursive: true, force: true })}
})
