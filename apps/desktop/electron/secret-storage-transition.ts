/** One explicit policy change across the Desktop's fixed credential stores. */
import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'

import { writeSecretFileAtomic } from './hardening'
import { SECRET_STORAGE_POLICY_FILE, type SecretStoragePolicy } from './secret-storage-policy'

export const SECRET_STORAGE_RECOVERY_FILE = 'secret-storage-recovery.json'
const STORES = ['connection.json', 'connections.json', 'native-oauth-tokens.json']
const ROOM = /^room-setup\/[0-9a-f-]{36}\.json$/
const STAGE_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.tmp$/
interface Change { name: string; before: Buffer | null; after: Buffer }
interface RecoveryEntry { name: string; before: string | null; afterDigest: string }
const digest = (value: Buffer) => createHash('sha256').update(value).digest('hex')
const same = (left: Buffer | null, right: Buffer | null) => left === null ? right === null : right !== null && left.equals(right)
const allowed = (name: string) => STORES.includes(name) || name === SECRET_STORAGE_POLICY_FILE || ROOM.test(name)

function directorySync(directory: string) {
  // Match the existing durable secret writer's platform boundary.
  if (process.platform === 'win32') {return}
  const fd = fs.openSync(directory, 'r')
  try {fs.fsyncSync(fd)} finally {fs.closeSync(fd)}
}

function read(file: string, privateFile = false, maxBytes = Infinity): Buffer | null {
  let stat: fs.Stats
  try {stat = fs.lstatSync(file)} catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {return null}
    throw error
  }
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > maxBytes ||
      (process.getuid && (stat.uid !== process.getuid() || (privateFile && (stat.mode & 0o077) !== 0)))) {
    throw new Error('Credential recovery requires private, owned regular files.')
  }
  return fs.readFileSync(file)
}

function parse(bytes: Buffer) {
  try {return JSON.parse(bytes.toString('utf8'))}
  catch {throw new Error('Credential storage is unreadable.')}
}

function roomDirectory(directory: string) {
  const stat = fs.lstatSync(path.join(directory, 'room-setup'))
  if (!stat.isDirectory() || stat.isSymbolicLink() ||
      (process.getuid && (stat.uid !== process.getuid() || (stat.mode & 0o077) !== 0))) {
    throw new Error('Room setup credential storage is unreadable.')
  }
}

function write(file: string, bytes: Buffer) {
  writeSecretFileAtomic(file, bytes, { uniqueStage: true, durable: {
    verify: actual => {if (!actual.equals(bytes)) {throw new Error('Credential write verification failed.')}}
  } })
}

function forget(directory: string, journal: Buffer) {
  const file = path.join(directory, SECRET_STORAGE_RECOVERY_FILE)
  try {
    fs.unlinkSync(file)
    directorySync(directory)
  } catch (error) {
    // If unlink succeeded but its flush did not, keep a recoverable decision.
    if (!fs.existsSync(file)) {write(file, journal)}
    throw error
  }
}

function stages(directory: string, name: string) {
  const folder = path.dirname(path.join(directory, name)), prefix = path.basename(name) + '.'
  try {
    return fs.readdirSync(folder).filter(entry => entry.startsWith(prefix) && STAGE_ID.test(entry.slice(prefix.length)))
      .map(entry => path.join(folder, entry))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {return []}
    throw error
  }
}

function retireCredentialStages(directory: string, expected?: Map<string, Set<string>>) {
  const names = [...STORES, SECRET_STORAGE_POLICY_FILE]
  if (fs.existsSync(path.join(directory, 'room-setup'))) {
    roomDirectory(directory)
    for (const entry of fs.readdirSync(path.join(directory, 'room-setup'))) {
      const name = 'room-setup/' + entry.slice(0, 41)
      if (ROOM.test(name) && !names.includes(name)) {names.push(name)}
    }
  }
  const owned: string[] = []
  for (const name of names) {
    for (const file of stages(directory, name)) {
      const bytes = read(file, true)!
      const hashes = expected?.get(name)
      if (hashes) {
        if (!hashes.has(digest(bytes))) {throw new Error('An interrupted credential stage needs recovery.')}
      } else {
        const value = parse(bytes)
        if (!value || typeof value !== 'object' || Array.isArray(value)) {
          throw new Error('An interrupted credential stage is unreadable.')
        }
      }
      owned.push(file)
    }
  }
  // Only this writer's complete, private UUID stages are unpublished. Legacy
  // fixed .tmp files and other names are never claimed or removed here.
  for (const file of owned) {fs.unlinkSync(file); directorySync(path.dirname(file))}
  return owned.length > 0
}

/** A recovery refusal must leave the user a way to retry or quit before boot. */
export async function recoverSecretStorageAtStartup(recover: () => void, retry: () => Promise<boolean>): Promise<boolean> {
  for (;;) {
    try {recover(); return true} catch {if (!await retry()) {return false}}
  }
}

/** Restore the previous bytes before policy reads; recovery never decrypts. */
export function recoverSecretStorageTransition(directory: string): boolean {
  const file = path.join(directory, SECRET_STORAGE_RECOVERY_FILE)
  const published = read(file, true)
  const prepared = stages(directory, SECRET_STORAGE_RECOVERY_FILE).map(file => ({ file, bytes: read(file, true)! }))
  const journal = published ?? prepared[0]?.bytes
  if (!journal) {return retireCredentialStages(directory)}
  if (prepared.some(stage => !stage.bytes.equals(journal))) {throw new Error('Credential recovery stages disagree.')}
  const saved = parse(journal)
  if (saved?.version !== 1 || Object.keys(saved).sort().join() !== 'entries,version' ||
      !Array.isArray(saved.entries) || !saved.entries.length) {throw new Error('Credential recovery record is unreadable.')}
  const entries: Array<{ name: string; before: Buffer | null; current: Buffer | null; afterDigest: string }> = []
  const names = new Set<string>()
  for (const entry of saved.entries as RecoveryEntry[]) {
    if (!entry || typeof entry.name !== 'string' || !allowed(entry.name) || names.has(entry.name) ||
        Object.keys(entry).sort().join() !== 'afterDigest,before,name' ||
        typeof entry.afterDigest !== 'string' || !/^[0-9a-f]{64}$/.test(entry.afterDigest) ||
        (entry.before !== null && typeof entry.before !== 'string') ||
        (entry.before === null && entry.name !== SECRET_STORAGE_POLICY_FILE)) {
      throw new Error('Credential recovery record is unreadable.')
    }
    names.add(entry.name)
    const before = entry.before === null ? null : Buffer.from(entry.before, 'base64')
    if (before !== null && before.toString('base64') !== entry.before) {throw new Error('Credential recovery bytes are unreadable.')}
    if (ROOM.test(entry.name)) {roomDirectory(directory)}
    const current = read(path.join(directory, entry.name), ROOM.test(entry.name), ROOM.test(entry.name) ? 65536 : Infinity)
    if (!same(current, before) && (current === null || digest(current) !== entry.afterDigest)) {
      throw new Error('Credential storage changed outside its pending recovery.')
    }
    entries.push({ name: entry.name, before, current, afterDigest: entry.afterDigest })
  }
  if (!names.has(SECRET_STORAGE_POLICY_FILE)) {throw new Error('Credential recovery policy is missing.')}
  // A crash can leave the verified before-image stage before its atomic rename.
  // Publish that exact validated record before attempting any restoration.
  if (published === null) {write(file, journal)}
  // Validate every target before restoring any of them. A damaged record never
  // becomes permission to overwrite a different connection/profile or file.
  for (const entry of entries) {
    const file = path.join(directory, entry.name)
    if (entry.before === null) {
      if (entry.current !== null) {fs.unlinkSync(file); directorySync(path.dirname(file))}
    } else {
      write(file, entry.before)
    }
  }
  retireCredentialStages(directory, new Map(entries.map(entry => [entry.name,
    new Set([entry.afterDigest, ...(entry.before === null ? [] : [digest(entry.before)])])
  ])))
  for (const stage of prepared) {fs.unlinkSync(stage.file); directorySync(directory)}
  forget(directory, journal)
  return true
}

function commit(directory: string, changes: Change[]) {
  const file = path.join(directory, SECRET_STORAGE_RECOVERY_FILE)
  const journal = Buffer.from(JSON.stringify({ version: 1, entries: changes.map(change => ({
    name: change.name, before: change.before?.toString('base64') ?? null, afterDigest: digest(change.after)
  })) }))
  // Nothing changes until the old bytes and exact target identities are durable.
  for (const change of changes) {
    if (!same(read(path.join(directory, change.name), ROOM.test(change.name)), change.before)) {
      throw new Error('Credential storage changed during conversion.')
    }
  }
  write(file, journal)
  try {
    for (const change of changes) {write(path.join(directory, change.name), change.after)}
    forget(directory, journal)
  } catch (error) {
    // A failed restoration leaves the durable recovery record in place. Every
    // next consumer must recover or refuse; caches cannot bypass this boundary.
    recoverSecretStorageTransition(directory)
    throw error
  }
}

export function changeSecretStorageEncryption(options: {
  directory: string; policy: SecretStoragePolicy; on: boolean
  available: () => boolean
  encrypt: (plaintext: string, room: boolean) => { encoding?: string; value?: string } | null
  decrypt: (secret: { encoding?: string; value?: string }) => string
}): SecretStoragePolicy {
  const { directory, on, policy } = options
  recoverSecretStorageTransition(directory)
  if (policy.on === on) {return policy}
  if (on && !options.available()) {throw new Error('OS keychain encryption is unavailable on this machine.')}
  const changes: Change[] = []
  const convert = (secret: any, room = false) => {
    if (on ? secret?.encoding !== 'plain' || !secret.value : secret?.encoding !== 'safeStorage') {return secret}
    const plaintext = on ? String(secret.value) : options.decrypt(secret)
    if (!plaintext) {throw new Error('A stored credential could not be decrypted; storage policy was not changed.')}
    const next = on ? options.encrypt(plaintext, room) : { encoding: 'plain', value: plaintext }
    if (!next || (on && next.encoding !== 'safeStorage') || options.decrypt(next) !== plaintext) {
      throw new Error('Credential conversion verification failed; storage policy was not changed.')
    }
    return next
  }
  const block = (value: any) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {return value}
    return { ...value, ...(value.token ? { token: convert(value.token) } : {}),
      ...(value.headers && typeof value.headers === 'object' ? {
        headers: Object.fromEntries(Object.entries(value.headers).map(([key, secret]) => [key, convert(secret)]))
      } : {}) }
  }
  const plan = (name: string, rewrite: (value: any) => any) => {
    const before = read(path.join(directory, name), ROOM.test(name), ROOM.test(name) ? 65536 : Infinity)
    if (before === null) {return}
    const current = parse(before)
    if (!current || typeof current !== 'object' || Array.isArray(current)) {throw new Error('Stored credentials are unreadable.')}
    const next = rewrite(current)
    if (JSON.stringify(next) === JSON.stringify(current)) {return}
    const after = Buffer.from(JSON.stringify(next, null, 2))
    if (ROOM.test(name) && after.length > 65536) {throw new Error('Converted room credential exceeds its storage limit.')}
    changes.push({ name, before, after })
  }
  plan(STORES[0], value => ({ ...value, remote: block(value.remote),
    ...(value.profiles ? { profiles: Object.fromEntries(Object.entries(value.profiles).map(([key, entry]) => [key, block(entry)])) } : {}) }))
  plan(STORES[1], value => ({ ...value, ...(value.connections ? { connections: value.connections.map(block) } : {}) }))
  plan(STORES[2], value => Object.fromEntries(Object.entries(value).map(([key, secret]) => [key, convert(secret)])))
  if (fs.existsSync(path.join(directory, 'room-setup'))) {
    roomDirectory(directory)
    for (const name of fs.readdirSync(path.join(directory, 'room-setup')).sort()) {
      if (ROOM.test('room-setup/' + name)) {plan('room-setup/' + name, value => convert(value, true))}
    }
  }
  const next = { on, migrated: true }
  changes.push({ name: SECRET_STORAGE_POLICY_FILE, before: read(path.join(directory, SECRET_STORAGE_POLICY_FILE)),
    after: Buffer.from(JSON.stringify(next)) })
  commit(directory, changes)
  return next
}
