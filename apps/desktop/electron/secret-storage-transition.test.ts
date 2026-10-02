import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

import { buildSync } from 'esbuild'
import { test, vi } from 'vitest'

import { requireRoomSetupEncryption, SECRET_STORAGE_POLICY_FILE } from './secret-storage-policy'
import { changeSecretStorageEncryption, recoverSecretStorageTransition, recoverSecretStorageAtStartup, SECRET_STORAGE_RECOVERY_FILE } from './secret-storage-transition'
import { roomSetupCoordinator } from './room-setup'
import { roomSetupStore } from './room-setup-store'

const ROOM = 'room-setup/01234567-89ab-cdef-0123-456789abcdef.json'
const TARGETS = ['connection.json', 'connections.json', 'native-oauth-tokens.json', ROOM, SECRET_STORAGE_POLICY_FILE]
const plain = (value: string) => ({ encoding: 'plain', value })
const encode = (value: string) => ({ encoding: 'safeStorage', value: Buffer.from('sealed:' + value).toString('base64') })
const decode = (secret: any) => secret.encoding === 'safeStorage'
  ? Buffer.from(secret.value, 'base64').toString().replace(/^sealed:/, '') : secret.value

function fixture() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'credential-transition-'))
  fs.mkdirSync(path.join(directory, 'room-setup'), { mode: 0o700 })
  const values = [
    { remote: { token: plain('remote') }, profiles: {
      alpha: { token: plain('alpha'), headers: { 'CF-Access-Client-Secret': plain('alpha-header') } },
      beta: { token: plain('beta') }
    } },
    { connections: [{ id: 'one', token: plain('one') }, { id: 'two', token: plain('two') }] },
    { 'https://one.invalid': plain('one-oauth'), 'https://two.invalid': plain('two-oauth') },
    plain(JSON.stringify({ id: path.basename(ROOM, '.json'), setupId: path.basename(ROOM, '.json'), kind: 'home',
      route: { connectionId: 'two', profile: 'beta' }, grant: 'held-grant', committed: true,
      installationId: 'installation', roomId: 'room' })),
    { on: false, migrated: true }
  ]
  TARGETS.forEach((name, index) => fs.writeFileSync(path.join(directory, name), JSON.stringify(values[index]), { mode: 0o600 }))
  const snapshot = () => TARGETS.map(name => fs.readFileSync(path.join(directory, name)))
  const options = { directory, policy: { on: false, migrated: true }, on: true,
    available: () => true, encrypt: encode, decrypt: decode }
  return { directory, snapshot, options }
}

test.each([...TARGETS, 'encryption', 'unavailable', 'basic_text'])(
  'failed transition at %s keeps the exact prior stores and policy, then retry preserves every owner', fault => {
    const p = fixture(), before = p.snapshot()
    const unrelated = path.join(p.directory, 'connections.json.tmp')
    fs.writeFileSync(unrelated, 'unrelated stage')
    const rename = fs.renameSync
    let failed = false, encryptions = 0
    const interceptor = vi.spyOn(fs, 'renameSync').mockImplementation((source, target) => {
      if (!failed && String(target) === path.join(p.directory, fault)) {
        failed = true; throw Object.assign(new Error('injected write refusal'), { code: 'EIO' })
      }
      rename(source, target)
    })
    try {
      assert.throws(() => changeSecretStorageEncryption({ ...p.options,
        available: () => {
          if (fault === 'basic_text') {requireRoomSetupEncryption({ on: true, migrated: true }, () => 'basic_text')}
          return fault !== 'unavailable'
        },
        encrypt: value => {
          if (fault === 'encryption' && ++encryptions === 2) {throw new Error('keychain refused encryption')}
          return encode(value)
        }
      }))
      assert.deepEqual(p.snapshot(), before)
      assert.equal(fs.existsSync(path.join(p.directory, SECRET_STORAGE_RECOVERY_FILE)), false)
      assert.equal(fs.readFileSync(unrelated, 'utf8'), 'unrelated stage')
      interceptor.mockRestore()
      assert.deepEqual(changeSecretStorageEncryption(p.options), { on: true, migrated: true })
      const converted = p.snapshot().map(bytes => JSON.parse(bytes.toString()))
      assert.equal(decode(converted[0].profiles.alpha.token), 'alpha')
      assert.equal(decode(converted[0].profiles.beta.token), 'beta')
      assert.equal(decode(converted[0].profiles.alpha.headers['CF-Access-Client-Secret']), 'alpha-header')
      assert.equal(decode(converted[1].connections[1].token), 'two')
      assert.equal(decode(converted[2]['https://one.invalid']), 'one-oauth')
      assert.equal(JSON.parse(decode(converted[3])).route.profile, 'beta')
      if (process.getuid) {
        for (const name of TARGETS) {assert.equal(fs.statSync(path.join(p.directory, name)).mode & 0o777, 0o600)}
      }
      const encrypted = p.snapshot()
      assert.throws(() => changeSecretStorageEncryption({ ...p.options, policy: { on: true, migrated: true }, on: false,
        decrypt: () => '' }))
      assert.deepEqual(p.snapshot(), encrypted)
      assert.deepEqual(changeSecretStorageEncryption({ ...p.options, policy: { on: true, migrated: true }, on: false }),
        { on: false, migrated: true })
      assert.deepEqual(p.snapshot().map(bytes => JSON.parse(bytes.toString())), before.map(bytes => JSON.parse(bytes.toString())))
    } finally {interceptor.mockRestore(); fs.rmSync(p.directory, { recursive: true, force: true })}
  }
)

test.each([... [SECRET_STORAGE_RECOVERY_FILE, ...TARGETS].flatMap(boundary =>
  ['before', 'after'].map(phase => [boundary, phase])), ['native-orphan', 'before']])(
  'a fresh process recovers a crash at %s %s without decrypting or losing its recovery record', async (boundary, phase) => {
    const p = fixture(), before = p.snapshot()
    const bundle = path.join(p.directory, 'transition.cjs')
    buildSync({ stdin: { contents: "export * from './secret-storage-transition'; export {writeSecretFileAtomic} from './hardening'",
      resolveDir: path.dirname(fileURLToPath(import.meta.url)), loader: 'ts' },
      outfile: bundle, bundle: true, platform: 'node', format: 'cjs', logLevel: 'silent' })
    const child = `
      const fs = require('node:fs'), path = require('node:path');
      const [bundle, directory, boundary, phase] = process.argv.slice(1);
      const { changeSecretStorageEncryption, writeSecretFileAtomic } = require(bundle);
      const rename = fs.renameSync;
      fs.renameSync = (source, target) => {
        const match = target === path.join(directory, boundary === 'native-orphan' ? 'native-oauth-tokens.json' : boundary);
        if (match && phase === 'before') process.exit(77);
        rename(source, target);
        if (match && phase === 'after') process.exit(77);
      };
      if (boundary === 'native-orphan') {
        const file = path.join(directory, 'native-oauth-tokens.json');
        const store = JSON.parse(fs.readFileSync(file));
        store['https://new.invalid'] = {encoding:'plain',value:'unacknowledged-token-set'};
        writeSecretFileAtomic(file, JSON.stringify(store), {uniqueStage:true,durable:{verify:()=>{}}});
      }
      changeSecretStorageEncryption({ directory, policy: {on:false,migrated:true}, on:true,
        available: () => true,
        encrypt: value => ({encoding:'safeStorage', value:Buffer.from('sealed:' + value).toString('base64')}),
        decrypt: secret => secret.encoding === 'safeStorage'
          ? Buffer.from(secret.value,'base64').toString().replace(/^sealed:/,'') : secret.value
      });
    `
    try {
      const result = spawnSync(process.execPath, ['-e', child, bundle, p.directory, boundary, phase], { encoding: 'utf8' })
      assert.equal(result.status, 77, result.stderr)
      const journalFile = path.join(p.directory, SECRET_STORAGE_RECOVERY_FILE)
      if (boundary === 'native-orphan') {
        assert.equal(fs.existsSync(journalFile), false)
        assert.equal(recoverSecretStorageTransition(p.directory), true)
        assert.deepEqual(p.snapshot(), before)
        assert.deepEqual(changeSecretStorageEncryption(p.options), { on: true, migrated: true })
        assert.equal(fs.readdirSync(p.directory).some(name => name.endsWith('.tmp')), false)
        return
      }
      if (boundary === SECRET_STORAGE_RECOVERY_FILE && phase === 'before') {
        assert.equal(fs.existsSync(journalFile), false)
        assert.equal(recoverSecretStorageTransition(p.directory), true)
        assert.deepEqual(p.snapshot(), before)
        assert.equal(fs.readdirSync(p.directory).some(name => name.endsWith('.tmp')), false)
        assert.deepEqual(changeSecretStorageEncryption(p.options), { on: true, migrated: true })
        return
      }
      const journal = fs.readFileSync(journalFile)
      const interrupted = p.snapshot()
      if (process.getuid) {assert.equal(fs.statSync(journalFile).mode & 0o777, 0o600)}
      // Unknown paths never become write authority, even in a private manifest.
      const tampered = JSON.parse(journal.toString())
      tampered.entries.at(-1).name = '../unrelated.json'
      fs.writeFileSync(journalFile, JSON.stringify(tampered))
      assert.throws(() => recoverSecretStorageTransition(p.directory), /unreadable/)
      assert.equal(fs.existsSync(journalFile), true)
      assert.deepEqual(p.snapshot(), interrupted)
      fs.writeFileSync(journalFile, 'disposable-secret-not-json')
      assert.throws(() => recoverSecretStorageTransition(p.directory), error =>
        error instanceof Error && error.message === 'Credential storage is unreadable.')
      assert.deepEqual(p.snapshot(), interrupted)
      let prompts = 0
      assert.equal(await recoverSecretStorageAtStartup(() => recoverSecretStorageTransition(p.directory), async () => {
        prompts++; return false
      }), false)
      assert.equal(prompts, 1)
      assert.equal(fs.readFileSync(journalFile, 'utf8'), 'disposable-secret-not-json')
      fs.writeFileSync(journalFile, journal)
      const rename = fs.renameSync
      const refusal = vi.spyOn(fs, 'renameSync').mockImplementation((source, target) => {
        if (String(target) === path.join(p.directory, 'connection.json')) {throw new Error('recovery unavailable')}
        rename(source, target)
      })
      try {
        assert.throws(() => recoverSecretStorageTransition(p.directory), /recovery unavailable/)
        assert.deepEqual(fs.readFileSync(journalFile), journal)
        let connections = 0
        const coordinator = roomSetupCoordinator({
          beforeOperation: () => {recoverSecretStorageTransition(p.directory)},
          store: roomSetupStore({ directory: path.join(p.directory, 'room-setup'),
            encrypt: text => JSON.stringify(plain(text)), decrypt: text => decode(JSON.parse(text)) }),
          connect: async () => {connections++; throw new Error('must not connect')}
        })
        await assert.rejects(coordinator.recover(), /recovery unavailable/)
        assert.equal(connections, 0)
        assert.equal(fs.existsSync(path.join(p.directory, ROOM)), true)
      } finally {refusal.mockRestore()}
      // Recovery has no encryption/decryption callbacks at all. Reopening only
      // restores verified raw bytes before the normal OFF reader can run.
      assert.equal(recoverSecretStorageTransition(p.directory), true)
      assert.deepEqual(p.snapshot(), before)
      assert.equal(recoverSecretStorageTransition(p.directory), false)
      assert.deepEqual(changeSecretStorageEncryption(p.options), { on: true, migrated: true })
      assert.equal(fs.readdirSync(p.directory).some(name => name.endsWith('.tmp')), false)
      assert.equal(fs.readdirSync(path.join(p.directory, 'room-setup')).some(name => name.endsWith('.tmp')), false)
    } finally {fs.rmSync(p.directory, { recursive: true, force: true })}
  }
)
