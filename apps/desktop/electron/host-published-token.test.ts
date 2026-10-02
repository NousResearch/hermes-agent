import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { attachToHostBackend } from './host-backend-attach'
import { lookupPublishedSessionToken, publishedTokenForRecord } from './host-published-token'

const RECORD = {
  createTime: 1_000,
  host: '127.0.0.1',
  pid: 4711,
  port: 65_238,
  profile: 'ops',
  purpose: 'serve',
  registeredAt: 2_000
}

function fingerprint(token: string): string {
  return createHash('sha256').update(token, 'utf8').digest('hex').slice(0, 16)
}

function publication(role: string, token: string, overrides: Record<string, unknown> = {}) {
  return {
    recordText: JSON.stringify({
      createTime: 1_000,
      host: '127.0.0.1',
      pid: 4711,
      port: 65_238,
      protocolVersion: 1,
      role,
      tokenFingerprint: fingerprint(token),
      ...overrides
    }),
    role,
    token
  }
}

test('a desktop-serve publication matches the ledger record by pid, port, and fingerprint', () => {
  const token = 'published-session-token'

  assert.equal(publishedTokenForRecord(RECORD, [publication('desktop-serve', token)]), token)
  assert.equal(
    publishedTokenForRecord(RECORD, [publication('desktop-serve', token, { tokenFingerprint: '0'.repeat(16) })]),
    null
  )
  assert.equal(publishedTokenForRecord(RECORD, [publication('desktop-serve', token, { pid: 99 })]), null)
  assert.equal(publishedTokenForRecord(RECORD, [publication('gateway', token)]), null)
})

test('lookup adopts an owner-only desktop-serve token and ignores a world-readable one', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-host-token-'))
  const token = 'published-session-token'

  const record = {
    createTime: 1_000,
    host: '127.0.0.1',
    pid: 4711,
    port: 65_238,
    protocolVersion: 1,
    role: 'desktop-serve',
    tokenFingerprint: fingerprint(token)
  }

  fs.chmodSync(directory, 0o700)
  fs.writeFileSync(path.join(directory, 'host-desktop-serve.json'), JSON.stringify(record), { mode: 0o600 })
  fs.writeFileSync(path.join(directory, 'host-desktop-serve.token'), `${token}\n`, { mode: 0o600 })

  const env = { home: os.homedir(), lockDir: directory, platform: process.platform }

  const io = {
    lstat: (target: string) => fs.lstatSync(target),
    readFile: (target: string) => fs.readFileSync(target, 'utf8'),
    uid: typeof process.getuid === 'function' ? process.getuid() : null
  }

  assert.equal(lookupPublishedSessionToken(RECORD, env, io), token)

  fs.chmodSync(path.join(directory, 'host-desktop-serve.token'), 0o644)
  assert.equal(process.platform === 'win32' ? token : null, lookupPublishedSessionToken(RECORD, env, io))
  fs.rmSync(directory, { recursive: true, force: true })
})

test('a withheld dashboard token adopts the on-disk published token instead of spawning', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-host-token-'))
  const token = 'published-session-token'

  fs.chmodSync(directory, 0o700)
  fs.writeFileSync(
    path.join(directory, 'host-desktop-serve.json'),
    JSON.stringify({
      host: '127.0.0.1',
      pid: 4711,
      port: 65_238,
      protocolVersion: 1,
      role: 'desktop-serve',
      tokenFingerprint: fingerprint(token)
    }),
    { mode: 0o600 }
  )
  fs.writeFileSync(path.join(directory, 'host-desktop-serve.token'), token, { mode: 0o600 })

  const logs: string[] = []

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    {
      log: message => {
        logs.push(message)
      },
      probeWebSocket: async wsUrl => {
        assert.match(wsUrl, /token=published-session-token/)

        return { ok: true }
      },
      publishedTokenFor: record =>
        lookupPublishedSessionToken(
          record,
          { home: os.homedir(), lockDir: directory, platform: process.platform },
          {
            lstat: target => fs.lstatSync(target),
            readFile: target => fs.readFileSync(target, 'utf8'),
            uid: typeof process.getuid === 'function' ? process.getuid() : null
          }
        ),
      readLedger: () =>
        JSON.stringify([
          {
            host: '127.0.0.1',
            pid: 4711,
            port: 65_238,
            profile: 'ops',
            purpose: 'dashboard',
            registered_at: 2_000
          }
        ]),
      resolveServedToken: async () => null,
      waitForReady: async () => undefined
    }
  )

  assert.equal(attached?.token, token)
  assert.equal(
    logs.some(line => line.includes('did not publish a session token')),
    false
  )
  fs.rmSync(directory, { recursive: true, force: true })
})
