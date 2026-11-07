/**
 * gateway-connections-store.ts
 *
 * The machine's ONE saved-gateways list, shared by native Desktop (this module)
 * and every browser on this machine's Webapp (the Python twin
 * `hermes_cli/gateway_connections.py`, over `/api/gateway-connections`).
 *
 * File: `<root>/gateway-connections.json` beside `ui-preferences.json` (same
 * root rule), mode 0600. Document = the v2 registry of connection-registry.ts
 * plus `revision`, `importedFrom` (the userData directories already merged in)
 * and `importNotices`. Same lock contract as ui-preferences.ts: the
 * `<store>.lock` directory (mkdir is atomic) with an `owner` token, 3 s wait,
 * 10 s stale break, ownership re-checked before the temp+rename replace, an
 * unparseable file kept aside as `.corrupt-<ms>`, and only ENOENT reads as "no
 * store". The lock here is synchronous because main's registry reads and writes
 * are (they sit inside sync IPC paths); a write holds it for milliseconds.
 *
 * Secrets in the file are `{encoding: 'plain'}` envelopes. Every write passes
 * them through `decrypt` once, so a safeStorage envelope main produced (or the
 * old userData registry held) lands as plain; one that will not decrypt stays
 * as its original envelope and is named in `importNotices`, never dropped.
 *
 * OAuth sign-ins: `<root>/gateway-connection-tokens.json`
 * `{version: 1, revision, tokens: {<connection id>: BoundTokenSet},
 *   unbound: {<base url>: NativeTokenSet}, undecrypted: {<base url>: envelope}}`.
 * A connection's set carries the `baseUrl` that issued it: it is handed out
 * only while the entry still names that URL, and a write that removes the
 * entry or changes its url/authMode drops it (ids are recycled label slugs).
 * `unbound` holds a sign-in for a URL no saved entry names yet (a pre-save
 * login, the v1 global remote); `undecrypted` keeps imported blobs the keychain
 * would not open.
 *
 * Migration: the first read on a userData directory not yet in `importedFrom`
 * merges that directory's registry (after the v1→v2 migration and the drift
 * reconcile, in memory) and its native-oauth-tokens.json into the store, by id
 * and then by the URL / SSH-target duplicate rule. An existing store is never
 * overwritten; the userData files are never written.
 */
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { normalizeRemoteBaseUrl } from './connection-config'
import {
  connectionIdForLabel,
  type ConnectionRegistry,
  LOCAL_CONNECTION_ID,
  migrateV1ToRegistry,
  normalizeRegistry,
  reconcileRegistryDrift,
  REGISTRY_QUARANTINE_CAP,
  type RegistryConnection,
  uniqueLabel
} from './connection-registry'
import type { NativeTokenSet } from './native-oauth'
import type { NativeTokenStoreIo } from './native-token-store'

export const GATEWAY_CONNECTIONS_FILE = 'gateway-connections.json'
export const GATEWAY_CONNECTION_TOKENS_FILE = 'gateway-connection-tokens.json'
const LOCK_WAIT_MS = 3000
const LOCK_STALE_MS = 10_000
const LOCK_POLL_MS = 20
const WATCH_INTERVAL_MS = 1500
const DEFAULT_LOCAL_LABEL = 'This device'

export interface SecretEnvelope {
  encoding?: string
  value?: string
}

export interface ImportNotice {
  connectionId: string
  field: string
  reason: string
}

export interface ConnectionsChange {
  connectionId: string
  reason: 'removed' | 'saved' | 'updated'
}

export interface GatewayConnectionsStoreDeps {
  /** The store directory (main's ROOT Hermes home, through symlinks). */
  root: string
  /** Electron userData: where the pre-store connections.json / native-oauth-tokens.json live. */
  userData: string
  /** The userData registry after main's v1→v2 migration and drift reconcile, computed in memory. */
  legacyRegistry: () => ConnectionRegistry
  /** The v1 connection.json, for the per-read drift reconcile main already runs. */
  reconcile?: (registry: ConnectionRegistry) => { changed: boolean; registry: ConnectionRegistry }
  /** decryptDesktopSecret: '' when the envelope cannot be opened. */
  decrypt: (secret: unknown) => string
  log?: (message: string) => void
}

interface StoreDoc extends Record<string, unknown> {
  revision?: number
  importedFrom?: Record<string, { at: number }>
  /** Quarantined entries past REGISTRY_QUARANTINE_CAP from a merge, carried verbatim (never dropped). */
  quarantinedOverflow?: unknown[]
}

/** The quarantined entries a merge of `store` and `incoming` keeps past the normalizer's cap. */
export function quarantineOverflow(store: ConnectionRegistry, incoming: ConnectionRegistry): unknown[] {
  return [...(store.quarantined || []), ...(incoming.quarantined || [])].slice(REGISTRY_QUARANTINE_CAP)
}

/** A connection's sign-in, bound to the gateway URL that issued it. */
export type BoundTokenSet = NativeTokenSet & { baseUrl: string }

interface TokensDoc {
  version: 1
  revision: number
  tokens: Record<string, BoundTokenSet>
  unbound: Record<string, NativeTokenSet>
  undecrypted: Record<string, SecretEnvelope>
}

// ── files and the lock ─────────────────────────────────────────────────────

const sleepSync = (ms: number) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)

interface DocRead<T> {
  corrupt: boolean
  doc: null | T
}

function readDoc<T>(file: string): DocRead<T> {
  let raw: string

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return { corrupt: false, doc: null }
    }

    throw new Error(`gateway connections store is unreadable: ${(error as Error).message}`)
  }

  try {
    const doc = JSON.parse(raw.replace(/^\uFEFF/, '')) as unknown

    if (doc && typeof doc === 'object' && !Array.isArray(doc)) {
      return { corrupt: false, doc: doc as T }
    }
  } catch {
    // Unparseable: no store; the next write keeps it aside first.
  }

  return { corrupt: true, doc: null }
}

const ownerFile = (lockDir: string) => path.join(lockDir, 'owner')

function owns(lockDir: string, token: string): boolean {
  try {
    return fs.readFileSync(ownerFile(lockDir), 'utf8') === token
  } catch {
    return false
  }
}

function breakIfStale(lockDir: string): void {
  try {
    if (Date.now() - fs.statSync(lockDir).mtimeMs > LOCK_STALE_MS) {
      fs.rmSync(lockDir, { force: true, recursive: true })
    }
  } catch {
    // Gone already, or just re-taken: the next mkdir decides.
  }
}

/** The cross-process lock shared with the Python writer (ui_preferences contract). */
export function withStoreLockSync<T>(file: string, run: (stillOurs: () => boolean) => T): T {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const lockDir = `${file}.lock`
  const token = `${process.pid}-${randomBytes(8).toString('hex')}`
  const deadline = Date.now() + LOCK_WAIT_MS

  for (;;) {
    try {
      fs.mkdirSync(lockDir)

      break
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') {
        throw error
      }

      breakIfStale(lockDir)

      if (Date.now() > deadline) {
        throw new Error('gateway connections store is busy')
      }

      sleepSync(LOCK_POLL_MS)
    }
  }

  try {
    fs.writeFileSync(ownerFile(lockDir), token, 'utf8')

    return run(() => owns(lockDir, token))
  } finally {
    if (owns(lockDir, token)) {
      fs.rmSync(lockDir, { force: true, recursive: true })
    }
  }
}

function replaceDoc(file: string, doc: unknown, corrupt: boolean, stillOurs: () => boolean): void {
  if (!stillOurs()) {
    // Broken as stale and re-taken: replacing now could drop the other writer's save.
    throw new Error('gateway connections store is busy')
  }

  if (corrupt) {
    fs.copyFileSync(file, `${file}.corrupt-${Date.now()}`)
  }

  const temp = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${Date.now()}.tmp`)

  try {
    fs.writeFileSync(temp, `${JSON.stringify(doc, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
    fs.renameSync(temp, file)
  } catch (error) {
    fs.rmSync(temp, { force: true })

    throw error
  }
}

const revisionOf = (doc: null | { revision?: unknown }) =>
  Number.isInteger(doc?.revision) ? (doc!.revision as number) : 0

// ── secrets ────────────────────────────────────────────────────────────────

export const plainSecretEnvelope = (value: unknown): null | SecretEnvelope => {
  const raw = String(value || '')

  return raw ? { encoding: 'plain', value: raw } : null
}

/** Every token/header envelope as plain; one `decrypt` cannot open stays as it was, with a notice. */
export function plainSecrets(
  registry: ConnectionRegistry,
  decrypt: (secret: unknown) => string
): { notices: ImportNotice[]; registry: ConnectionRegistry } {
  const notices: ImportNotice[] = []

  const convert = (secret: unknown, connectionId: string, field: string) => {
    if (!secret || typeof secret !== 'object' || (secret as SecretEnvelope).encoding === 'plain') {
      return secret
    }

    const value = decrypt(secret)

    if (value) {
      return { encoding: 'plain', value }
    }

    notices.push({ connectionId, field, reason: 'undecryptable' })

    return secret
  }

  const withPlainSecrets = <T extends { headers?: unknown; token?: unknown }>(entry: T, id: string): T => {
    const next = { ...entry }

    if (entry.token !== undefined) {
      next.token = convert(entry.token, id, 'token')
    }

    if (entry.headers && typeof entry.headers === 'object' && !Array.isArray(entry.headers)) {
      next.headers = Object.fromEntries(
        Object.entries(entry.headers).map(([name, secret]) => [name, convert(secret, id, `headers.${name}`)])
      )
    }

    return next
  }

  const connections = registry.connections.map(entry => withPlainSecrets(entry, entry.id))

  // Quarantined entries are user data too: their secrets get the same treatment.
  const quarantined = registry.quarantined?.map((item, index) =>
    item.entry && typeof item.entry === 'object' && !Array.isArray(item.entry)
      ? { ...item, entry: withPlainSecrets(item.entry as Record<string, unknown>, `quarantined.${index}`) }
      : item
  )

  return { notices, registry: { ...registry, connections, ...(quarantined ? { quarantined } : {}) } }
}

// ── merge rules ────────────────────────────────────────────────────────────

const urlKey = (value: unknown) =>
  String(value || '')
    .trim()
    .replace(/\/+$/, '')
    .toLowerCase()

const sshKey = (c: RegistryConnection) =>
  `${(c.user || '').toLowerCase()}@${(c.host || '').toLowerCase()}:${c.port ?? 22}::${(c.remoteProfile || '').trim()}`

/** The duplicate identity normalizeConnectionInput refuses twice: URL for remote/cloud, target for ssh. */
export function dialIdentity(c: RegistryConnection): null | string {
  if (c.kind === 'remote' || c.kind === 'cloud') {
    return `url:${urlKey(c.url)}`
  }

  return c.kind === 'ssh' ? `ssh:${sshKey(c)}` : null
}

function fillMissingSecrets(target: RegistryConnection, source: RegistryConnection): RegistryConnection {
  const next = { ...target }

  if (next.token === undefined && source.token !== undefined) {
    next.token = source.token
  }

  if (source.headers) {
    next.headers = { ...source.headers, ...(next.headers || {}) }
  }

  return next
}

/**
 * Merge `incoming` (Desktop's userData registry) into `store` without losing
 * either: same id + same dial identity is one connection (the store's copy
 * wins, missing secrets are filled), a dial-identity match under another id is
 * the same connection too, anything else is added (id and label made unique).
 * Desktop's primary / launchMode / lastUsed are adopted wherever the store's
 * is still the default (a Webapp-created store never chose one).
 */
export function mergeRegistries(store: ConnectionRegistry, incoming: ConnectionRegistry): ConnectionRegistry {
  const connections = store.connections.map(c => ({ ...c }))
  const idMap = new Map<string, string>([[LOCAL_CONNECTION_ID, LOCAL_CONNECTION_ID]])

  for (const entry of incoming.connections) {
    if (entry.kind === 'local') {
      const local = connections.find(c => c.kind === 'local')

      if (local && local.label === DEFAULT_LOCAL_LABEL && entry.label !== DEFAULT_LOCAL_LABEL) {
        local.label = uniqueLabel(
          entry.label,
          connections.filter(c => c !== local).map(c => c.label)
        )
      }

      continue
    }

    const identity = dialIdentity(entry)
    const sameId = connections.findIndex(c => c.id === entry.id && dialIdentity(c) === identity)

    const sameDial =
      sameId >= 0 ? sameId : connections.findIndex(c => identity !== null && dialIdentity(c) === identity)

    if (sameDial >= 0) {
      connections[sameDial] = fillMissingSecrets(connections[sameDial], entry)
      idMap.set(entry.id, connections[sameDial].id)

      continue
    }

    const taken = connections.map(c => c.id)

    const label = uniqueLabel(
      entry.label,
      connections.map(c => c.label)
    )

    const id = taken.includes(entry.id) ? connectionIdForLabel(label, taken) : entry.id
    connections.push({ ...entry, id, label })
    idMap.set(entry.id, id)
  }

  const quarantined = [...(store.quarantined || []), ...(incoming.quarantined || [])].slice(0, REGISTRY_QUARANTINE_CAP)
  const mapId = (id: string) => idMap.get(id) || LOCAL_CONNECTION_ID
  // A store value still at its default is no one's choice yet; Desktop's is.
  const primary = store.primary === LOCAL_CONNECTION_ID ? mapId(incoming.primary) : store.primary
  const lastUsed = store.lastUsed === store.primary ? mapId(incoming.lastUsed) : store.lastUsed

  return normalizeRegistry({
    ...store,
    connections,
    primary,
    launchMode: store.launchMode === 'primary' ? incoming.launchMode : store.launchMode,
    lastUsed,
    ...(quarantined.length ? { quarantined } : {})
  })
}

const sameJson = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null)

/**
 * Re-apply what a caller changed (`base` → `next`) on top of a store another
 * writer moved meanwhile (`disk`): changed/added entries upsert, entries the
 * caller removed go, scalars the caller changed win, everything else is disk's.
 */
export function rebaseRegistry(
  base: ConnectionRegistry,
  next: ConnectionRegistry,
  disk: ConnectionRegistry
): ConnectionRegistry {
  const baseById = new Map(base.connections.map(c => [c.id, c]))
  const nextIds = new Set(next.connections.map(c => c.id))
  let connections = disk.connections.filter(c => !(baseById.has(c.id) && !nextIds.has(c.id)))

  for (const entry of next.connections) {
    if (!sameJson(baseById.get(entry.id), entry)) {
      connections = connections.some(c => c.id === entry.id)
        ? connections.map(c => (c.id === entry.id ? entry : c))
        : [...connections, entry]
    }
  }

  const pick = <K extends 'lastUsed' | 'launchMode' | 'primary' | 'quarantined'>(key: K) =>
    sameJson(base[key], next[key]) ? disk[key] : next[key]

  return normalizeRegistry({
    ...disk,
    connections,
    primary: pick('primary'),
    launchMode: pick('launchMode'),
    lastUsed: pick('lastUsed'),
    quarantined: pick('quarantined')
  })
}

/** What another writer's save means for this process: per connection id. */
export function registryChanges(before: ConnectionRegistry, after: ConnectionRegistry): ConnectionsChange[] {
  const afterById = new Map(after.connections.map(c => [c.id, c]))
  const changes: ConnectionsChange[] = []

  for (const entry of before.connections) {
    const now = afterById.get(entry.id)

    if (!now) {
      changes.push({ connectionId: entry.id, reason: 'removed' })
    } else if (!sameJson({ ...entry, label: '' }, { ...now, label: '' })) {
      changes.push({ connectionId: entry.id, reason: 'updated' })
    } else if (entry.label !== now.label) {
      changes.push({ connectionId: entry.id, reason: 'saved' })
    }
  }

  for (const entry of after.connections) {
    if (!before.connections.some(c => c.id === entry.id)) {
      changes.push({ connectionId: entry.id, reason: 'saved' })
    }
  }

  if (
    !changes.length &&
    (before.primary !== after.primary || before.launchMode !== after.launchMode || before.lastUsed !== after.lastUsed)
  ) {
    changes.push({ connectionId: after.primary, reason: 'saved' })
  }

  return changes
}

// ── native OAuth token sets ─────────────────────────────────────────────────

function emptyTokens(doc: null | Partial<TokensDoc>): TokensDoc {
  const section = (key: keyof TokensDoc) =>
    doc && doc[key] && typeof doc[key] === 'object' && !Array.isArray(doc[key]) ? (doc[key] as any) : {}

  return {
    version: 1,
    revision: revisionOf(doc),
    tokens: section('tokens'),
    unbound: section('unbound'),
    undecrypted: section('undecrypted')
  }
}

const safeUrl = (url: unknown) => {
  try {
    return normalizeRemoteBaseUrl(url)
  } catch {
    return null
  }
}

function connectionIdForUrl(registry: ConnectionRegistry, baseUrl: string): null | string {
  const key = safeUrl(baseUrl)

  return (
    registry.connections.find(c => (c.kind === 'remote' || c.kind === 'cloud') && key && safeUrl(c.url) === key)?.id ||
    null
  )
}

interface LegacyTokenEntry {
  secret: SecretEnvelope
  set: NativeTokenSet | null
}

/** userData's native-oauth-tokens.json (baseUrl → safeStorage/plain envelope), each opened once. */
export function decryptLegacyTokens(
  userData: string,
  decrypt: (secret: unknown) => string
): Record<string, LegacyTokenEntry> {
  let legacy: Record<string, SecretEnvelope> = {}

  try {
    legacy = readDoc<Record<string, SecretEnvelope>>(path.join(userData, 'native-oauth-tokens.json')).doc || {}
  } catch {
    legacy = {}
  }

  return Object.fromEntries(
    Object.entries(legacy).map(([baseUrl, secret]) => {
      let set: NativeTokenSet | null = null

      try {
        set = JSON.parse(decrypt(secret)) as NativeTokenSet
      } catch {
        set = null
      }

      return [baseUrl, { secret, set: set?.accessToken ? set : null }]
    })
  )
}

/** The ids whose sign-in no longer belongs to them: removed, or re-pointed (url/authMode). */
export function signedInIdsGone(before: ConnectionRegistry, after: ConnectionRegistry): string[] {
  const now = new Map(after.connections.map(c => [c.id, c]))

  return before.connections
    .filter(c => {
      const next = now.get(c.id)

      return !next || next.url !== c.url || next.authMode !== c.authMode
    })
    .map(c => c.id)
}

/** A connection's set as main's loader reads it, or null when it was issued for another URL. */
function boundSet(set: BoundTokenSet, entryUrl: unknown): NativeTokenSet | null {
  const key = safeUrl(entryUrl)

  if (!key || safeUrl(set?.baseUrl) !== key) {
    return null
  }

  const { baseUrl: _baseUrl, ...rest } = set

  return rest
}

function importLegacyTokens(
  legacy: Record<string, LegacyTokenEntry>,
  registry: ConnectionRegistry,
  tokens: TokensDoc
): number {
  let imported = 0

  for (const [baseUrl, { secret, set }] of Object.entries(legacy)) {
    const id = connectionIdForUrl(registry, baseUrl)

    if ((id && tokens.tokens[id]) || tokens.unbound[baseUrl] || tokens.undecrypted[baseUrl]) {
      continue // the store already has a sign-in here: never overwrite it
    }

    if (!set) {
      tokens.undecrypted[baseUrl] = secret
    } else if (id) {
      // main's file was keyed by this URL: the set keeps it, so the sign-in still works here.
      tokens.tokens[id] = { ...set, baseUrl: safeUrl(baseUrl)! }
    } else {
      tokens.unbound[baseUrl] = set
    }

    imported += 1
  }

  return imported
}

/**
 * What Desktop's userData registry holds today, computed exactly as main did
 * before this store: connections.json normalized (or, when it does not exist,
 * the v1 connection.json migrated), then healed against v1 drift. Read-only:
 * the userData files stay as they are for older builds.
 */
export function readLegacyUserDataRegistry(
  userData: string,
  v1Config: unknown,
  log?: (message: string) => void
): ConnectionRegistry {
  const file = path.join(userData, 'connections.json')
  let registry: ConnectionRegistry

  try {
    const read = readDoc<Record<string, unknown>>(file)

    if (read.corrupt) {
      log?.(`[connections] ${file} could not be parsed; it is left in place and imports nothing.`)
    }

    registry = read.doc || read.corrupt ? normalizeRegistry(read.doc) : migrateV1ToRegistry(v1Config)
  } catch (error) {
    log?.(`[connections] ${(error as Error).message}; importing the v1 connection only.`)
    registry = migrateV1ToRegistry(v1Config)
  }

  return reconcileRegistryDrift(registry, v1Config).registry
}

// ── the store ──────────────────────────────────────────────────────────────

export interface GatewayConnectionsStore {
  file: string
  tokensFile: string
  read(): ConnectionRegistry
  write(registry: ConnectionRegistry): void
  invalidate(): void
  /** A NativeTokenStoreIo over the tokens file, keyed by connection id (unbound URLs fall back). */
  nativeTokenIo(log?: (message: string) => void): NativeTokenStoreIo
  /** Stat-poll both files; `onRegistry` gets per-connection changes another writer made. */
  watch(handlers: { onRegistry: (changes: ConnectionsChange[]) => void; onTokens: () => void }): () => void
}

export function createGatewayConnectionsStore(deps: GatewayConnectionsStoreDeps): GatewayConnectionsStore {
  const file = path.join(deps.root, GATEWAY_CONNECTIONS_FILE)
  const tokensFile = path.join(deps.root, GATEWAY_CONNECTION_TOKENS_FILE)

  const userDataKey = (() => {
    try {
      return fs.realpathSync.native(deps.userData)
    } catch {
      return path.resolve(deps.userData)
    }
  })()

  let cache: ConnectionRegistry | null = null
  let cacheStamp: null | string = null
  let cacheRevision = 0
  // The list as this process last wrote or announced it: the watcher reports
  // only what ANOTHER writer changed since, however often main re-read meanwhile.
  let announced: ConnectionRegistry | null = null

  const stampOf = () => {
    try {
      const stat = fs.statSync(file)

      return `${stat.mtimeMs}:${stat.size}:${stat.ino}`
    } catch {
      return 'missing'
    }
  }

  const remember = (registry: ConnectionRegistry, revision: number) => {
    announced = announced || registry
    cache = registry
    cacheRevision = revision
    cacheStamp = stampOf()

    return registry
  }

  /** Land `registry` (secrets made plain) on the locked doc; returns the stored registry. */
  const land = (
    doc: null | StoreDoc,
    corrupt: boolean,
    stillOurs: () => boolean,
    registry: ConnectionRegistry,
    extra: Partial<StoreDoc> = {}
  ) => {
    // Every write retries the envelopes still not plain, so `notices` is the whole current set.
    const { notices, registry: plain } = plainSecrets(registry, deps.decrypt)
    const revision = revisionOf(doc) + 1
    const overflow = [...(doc?.quarantinedOverflow || []), ...(extra.quarantinedOverflow || [])]

    if (overflow.length) {
      notices.push({ connectionId: '', field: 'quarantinedOverflow', reason: 'quarantine-cap' })
    }

    replaceDoc(
      file,
      {
        ...plain,
        revision,
        importedFrom: { ...(doc?.importedFrom || {}), ...(extra.importedFrom || {}) },
        ...(overflow.length ? { quarantinedOverflow: overflow } : {}),
        ...(notices.length ? { importNotices: notices } : {})
      },
      corrupt,
      stillOurs
    )

    announced = plain

    return remember(plain, revision)
  }

  const importUserData = () => {
    // Keychain reads (a macOS prompt can take seconds) happen before the lock is taken.
    const legacy = plainSecrets(deps.legacyRegistry(), deps.decrypt).registry
    const legacyTokens = decryptLegacyTokens(deps.userData, deps.decrypt)

    return withStoreLockSync(file, stillOurs => {
      const { corrupt, doc } = readDoc<StoreDoc>(file)

      if (doc?.importedFrom?.[userDataKey]) {
        return remember(normalizeRegistry(doc), revisionOf(doc))
      }

      const current = doc ? normalizeRegistry(doc) : null
      const merged = current ? mergeRegistries(current, legacy) : normalizeRegistry(legacy)
      const quarantinedOverflow = current ? quarantineOverflow(current, legacy) : []

      const tokens = withStoreLockSync(tokensFile, tokensOurs => {
        const read = readDoc<TokensDoc>(tokensFile)
        const next = emptyTokens(read.doc)

        if (importLegacyTokens(legacyTokens, merged, next)) {
          next.revision += 1
          replaceDoc(tokensFile, next, read.corrupt, tokensOurs)
        }

        return next
      })

      deps.log?.(
        `[connections] imported ${legacy.connections.length - 1} saved connection(s) and ${
          Object.keys(tokens.tokens).length + Object.keys(tokens.unbound).length
        } sign-in(s) from ${userDataKey} into ${file}`
      )

      return land(doc, corrupt, stillOurs, merged, {
        importedFrom: { [userDataKey]: { at: Date.now() } },
        quarantinedOverflow
      })
    })
  }

  const readFresh = (): ConnectionRegistry => {
    const { doc } = readDoc<StoreDoc>(file)

    if (!doc?.importedFrom?.[userDataKey]) {
      return importUserData()
    }

    const registry = remember(normalizeRegistry(doc), revisionOf(doc))
    const reconciled = deps.reconcile?.(registry)

    if (reconciled?.changed) {
      write(reconciled.registry)
    }

    return cache!
  }

  function read(): ConnectionRegistry {
    if (cache && cacheStamp === stampOf()) {
      return cache
    }

    try {
      return readFresh()
    } catch (error) {
      deps.log?.(`[connections] ${(error as Error).message}; keeping the last known list`)

      return cache || normalizeRegistry(null)
    }
  }

  function write(registry: ConnectionRegistry): void {
    const base = cache || read()
    const baseRevision = cacheRevision
    const plain = plainSecrets(registry, deps.decrypt).registry

    withStoreLockSync(file, stillOurs => {
      const { corrupt, doc } = readDoc<StoreDoc>(file)
      const disk = normalizeRegistry(doc)
      const moved = revisionOf(doc) !== baseRevision && base
      const next = moved ? rebaseRegistry(base, plain, disk) : normalizeRegistry(plain)

      const stored = land(doc, corrupt, stillOurs, next)
      dropTokens(signedInIdsGone(disk, stored))
    })
  }

  /** Forget the sign-ins kept under `ids` (store lock, then tokens lock: the import's order). */
  function dropTokens(ids: string[]): void {
    if (!ids.length || !fs.existsSync(tokensFile)) {
      return
    }

    withStoreLockSync(tokensFile, stillOurs => {
      const { corrupt, doc: raw } = readDoc<TokensDoc>(tokensFile)
      const doc = emptyTokens(raw)

      if (!ids.some(id => id in doc.tokens)) {
        return
      }

      for (const id of ids) {
        delete doc.tokens[id]
      }

      doc.revision += 1
      replaceDoc(tokensFile, doc, corrupt, stillOurs)
    })
  }

  function nativeTokenIo(log?: (message: string) => void): NativeTokenStoreIo {
    // native-token-store.ts serializes a baseUrl → envelope map; this adapter
    // maps it onto the id-keyed doc and writes only what that call changed.
    let lastRead: Record<string, SecretEnvelope> = {}

    const asMap = (doc: TokensDoc): Record<string, SecretEnvelope> => {
      const registry = read()
      const out: Record<string, SecretEnvelope> = {}

      // Kept as imported: the keychain may open them on a later launch.
      for (const [url, secret] of Object.entries(doc.undecrypted)) {
        out[url] = secret
      }

      for (const [url, set] of Object.entries(doc.unbound)) {
        out[url] = { encoding: 'plain', value: JSON.stringify(set) }
      }

      for (const [id, set] of Object.entries(doc.tokens)) {
        const url = registry.connections.find(c => c.id === id)?.url
        const key = safeUrl(url)
        const usable = boundSet(set, url)

        if (key && usable) {
          out[key] = { encoding: 'plain', value: JSON.stringify(usable) }
        }
      }

      return out
    }

    return {
      encrypt: plainSecretEnvelope,
      decrypt: secret =>
        (secret as SecretEnvelope)?.encoding === 'plain'
          ? String((secret as SecretEnvelope).value || '')
          : deps.decrypt(secret),
      readStoreText: () => {
        lastRead = asMap(emptyTokens(readDoc<TokensDoc>(tokensFile).doc))

        return JSON.stringify(lastRead)
      },
      writeStoreText: text => {
        const wanted = JSON.parse(text) as Record<string, SecretEnvelope>
        const registry = read()

        withStoreLockSync(tokensFile, stillOurs => {
          const { corrupt, doc: raw } = readDoc<TokensDoc>(tokensFile)
          const doc = emptyTokens(raw)

          for (const url of new Set([...Object.keys(lastRead), ...Object.keys(wanted)])) {
            if (sameJson(lastRead[url], wanted[url])) {
              continue
            }

            const id = connectionIdForUrl(registry, url)
            delete doc.unbound[url]
            delete doc.undecrypted[url]

            if (id) {
              delete doc.tokens[id]
            }

            if (wanted[url]) {
              // The Settings encryption toggle re-encodes through this io too: open it, never store it.
              const plain =
                wanted[url].encoding === 'plain' ? String(wanted[url].value || '') : deps.decrypt(wanted[url])

              const set = plain ? (JSON.parse(plain) as NativeTokenSet) : null

              if (!set) {
                doc.undecrypted[url] = wanted[url]
              } else if (id) {
                doc.tokens[id] = { ...set, baseUrl: safeUrl(url)! }
              } else {
                doc.unbound[url] = set
              }
            }
          }

          doc.revision += 1
          replaceDoc(tokensFile, doc, corrupt, stillOurs)
        })
        lastRead = wanted
      },
      rememberLog: log
    }
  }

  function watch(handlers: { onRegistry: (changes: ConnectionsChange[]) => void; onTokens: () => void }) {
    // No read here: main registers this at load, before safeStorage may be used.
    const onRegistryStat = () => {
      const before = announced
      const after = read()

      if (!before) {
        announced = after
      } else if (after !== before) {
        announced = after
        const changes = registryChanges(before, after)

        if (changes.length) {
          handlers.onRegistry(changes)
        }
      }
    }

    const onTokensStat = () => handlers.onTokens()

    fs.watchFile(file, { interval: WATCH_INTERVAL_MS, persistent: false }, onRegistryStat)
    fs.watchFile(tokensFile, { interval: WATCH_INTERVAL_MS, persistent: false }, onTokensStat)

    return () => {
      fs.unwatchFile(file, onRegistryStat)
      fs.unwatchFile(tokensFile, onTokensStat)
    }
  }

  return {
    file,
    tokensFile,
    read,
    write,
    invalidate: () => {
      cache = null
      cacheStamp = null
    },
    nativeTokenIo,
    watch
  }
}
