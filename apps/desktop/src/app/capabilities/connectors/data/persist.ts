import { type ProfileScope, profileScopeKey } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { readJson, writeJson } from '@/lib/storage'
import { $freeTierStatus } from '@/store/free-tier'

import { MCP_CATALOG_KEY } from '../../mcp/mcp-status'
import type { LocalServerInput } from '../types'

import { CONNECTOR_LIFETIMES, type ConnectorRead, CONNECTORS_QUERY_ROOT } from './keys'

export type PersistedRead = 'bundled' | 'servers' | ConnectorRead

const persists = (read: PersistedRead): boolean =>
  read === 'bundled' || read === 'servers' || CONNECTOR_LIFETIMES[read].persist

const STORAGE_PREFIX = 'hermes.connectors.v4.'

const IDENTITY_KEY = 'hermes.connectors.identity.v4'

const PERSIST_MAX_BYTES = 256 * 1024

type PersistedIdentity = 'guest' | 'signed-in'

interface PersistedEntry {
  at: number
  data: unknown
}

type PersistedValue = PersistedEntry['data']

interface PersistedBlob {
  bundled?: PersistedEntry
  catalog?: PersistedEntry
  identity: PersistedIdentity
  list?: PersistedEntry
  servers?: PersistedEntry
  tools?: Record<string, PersistedEntry>
}

const keyFor = (scopeKey: string) => `${STORAGE_PREFIX}${scopeKey}`

const identityOf = (hasGuest: boolean): PersistedIdentity => (hasGuest ? 'guest' : 'signed-in')

function currentIdentity(): PersistedIdentity | null {
  const status = $freeTierStatus.get()

  return status ? identityOf(status.has_guest) : readJson<PersistedIdentity>(IDENTITY_KEY)
}

const READS: ReadonlySet<PersistedValue> = new Set(Object.keys(CONNECTOR_LIFETIMES))

function size(value: PersistedBlob | PersistedEntry): number {
  try {
    return JSON.stringify(value)?.length ?? 0
  } catch {
    return Number.POSITIVE_INFINITY
  }
}

function readBlob(scopeKey: string): PersistedBlob | null {
  const blob = readJson<PersistedBlob>(keyFor(scopeKey))

  if (!(blob instanceof Object) || Array.isArray(blob)) {
    return null
  }

  return blob.identity === currentIdentity() ? blob : null
}

const SLOTS = {
  bundled: 'bundled',
  catalog: 'catalog',
  list: 'list',
  servers: 'servers'
} satisfies Partial<Record<PersistedRead, keyof PersistedBlob>>

type SlotRead = keyof typeof SLOTS

function entryOf(blob: PersistedBlob, read: PersistedRead, slug: string | undefined): PersistedEntry | undefined {
  if (!persists(read)) {
    return undefined
  }

  if (read === 'tools') {
    return slug ? blob.tools?.[slug] : undefined
  }

  if (!(read in SLOTS)) {
    return undefined
  }

  // SAFETY: guarded by `read in SLOTS` on the line above, and every slot holds a PersistedEntry.
  return blob[SLOTS[read as SlotRead]] as PersistedEntry | undefined
}

export interface QuerySeed<T> {
  initialData?: T
  initialDataUpdatedAt?: number
}

export function seedOptions<T>(scopeKey: ProfileScope, read: PersistedRead, slug?: string): QuerySeed<T> {
  if (currentIdentity() === null) {
    return {}
  }

  const blob = readBlob(profileScopeKey(scopeKey))
  const entry = blob ? entryOf(blob, read, slug) : undefined

  if (!entry || !Number.isFinite(entry.at) || entry.data === undefined || entry.data === null) {
    return {}
  }

  // SAFETY: the caller names the read whose answer it stored, so the entry holds that read's own result.
  return { initialData: entry.data as T, initialDataUpdatedAt: entry.at }
}

function trimmed(blob: PersistedBlob): PersistedBlob {
  let tools = blob.tools

  while (tools !== undefined && size({ ...blob, tools }) > PERSIST_MAX_BYTES) {
    const oldest = Object.entries(tools).sort((a, b) => a[1].at - b[1].at)[0]

    if (!oldest) {
      break
    }

    const rest = { ...tools }
    delete rest[oldest[0]]
    tools = rest
  }

  return { ...blob, tools }
}

function store(scopeKey: string, read: PersistedRead, slug: string | undefined, entry: PersistedEntry): void {
  const identity = currentIdentity()

  if (identity === null || !persists(read) || size(entry) > PERSIST_MAX_BYTES) {
    return
  }

  const blob = readBlob(scopeKey) ?? { identity }

  if (read === 'tools') {
    if (!slug) {
      return
    }

    blob.tools = { ...blob.tools, [slug]: entry }
  } else {
    // SAFETY: `read` is not 'tools' here, and every other PersistedRead has a slot.
    blob[SLOTS[read as SlotRead]] = entry
  }

  writeJson(keyFor(scopeKey), trimmed({ ...blob, identity }))
}

interface ReadTarget {
  read: PersistedRead
  scopeKey: string
  slug: string | undefined
}

/* oxlint-disable anti-slop/no-runtime-typeof -- SAFETY: a react-query key is typed `readonly unknown[]`; this function is the one boundary that parses one into a ReadTarget. */
function targetOf(queryKey: readonly unknown[]): null | ReadTarget {
  const [root, scopeKey, read, slug] = queryKey

  if (typeof scopeKey !== 'string') {
    return null
  }

  if (root === MCP_CATALOG_KEY[0]) {
    return { read: 'bundled', scopeKey, slug: undefined }
  }

  if (root !== CONNECTORS_QUERY_ROOT || !READS.has(read)) {
    return null
  }

  // SAFETY: READS holds exactly the CONNECTOR_LIFETIMES keys, and `read` is one of them by the check above.
  return { read: read as ConnectorRead, scopeKey, slug: typeof slug === 'string' ? slug : undefined }
}
/* oxlint-enable anti-slop/no-runtime-typeof */

const WRITE_DELAY_MS = 500

interface PendingWrite extends ReadTarget {
  entry: PersistedEntry
}

export function startConnectorPersistence(): () => void {
  const pending = new Map<string, PendingWrite>()
  const written = new Map<string, number>()
  let timer: ReturnType<typeof setTimeout> | null = null

  const flush = () => {
    timer = null

    for (const [key, write] of pending) {
      store(write.scopeKey, write.read, write.slug, write.entry)
      written.set(key, write.entry.at)
    }

    pending.clear()
  }

  const stopCache = queryClient.getQueryCache().subscribe(event => {
    if (event.type !== 'updated' || event.action.type !== 'success') {
      return
    }

    const target = targetOf(event.query.queryKey)
    const { data, dataUpdatedAt } = event.query.state

    if (!target || data === undefined) {
      return
    }

    const key = `${target.scopeKey}\u0000${target.read}\u0000${target.slug ?? ''}`

    if (written.get(key) === dataUpdatedAt) {
      return
    }

    pending.set(key, { ...target, entry: { at: dataUpdatedAt, data } })
    timer ??= setTimeout(flush, WRITE_DELAY_MS)
  })

  const stopIdentity = $freeTierStatus.listen(status => {
    const identity = status ? identityOf(status.has_guest) : null

    if (identity !== null && identity !== readJson<PersistedIdentity>(IDENTITY_KEY)) {
      writeJson(IDENTITY_KEY, identity)
    }
  })

  return () => {
    stopCache()
    stopIdentity()

    if (timer !== null) {
      clearTimeout(timer)
      flush()
    }
  }
}

interface ServerSeed {
  enabled: boolean
  name: string
}

function isSeed(value: PersistedValue): value is ServerSeed {
  if (value === null || !(value instanceof Object)) {
    return false
  }

  // SAFETY: an object here; the two field checks below are what make it a ServerSeed.
  const seed = value as Partial<ServerSeed>

  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- SAFETY: `seed` is a value read back from localStorage; this line is where it becomes a ServerSeed.
  return typeof seed.name === 'string' && typeof seed.enabled === 'boolean'
}

export function storeLocalServers(scope: ProfileScope, servers: readonly LocalServerInput[]): void {
  const seeds: ServerSeed[] = servers.map(({ enabled, name }) => ({ enabled, name }))

  store(profileScopeKey(scope), 'servers', undefined, { at: Date.now(), data: seeds })
}

export function seedLocalServers(scope: ProfileScope): LocalServerInput[] {
  const { initialData } = seedOptions<unknown>(scope, 'servers')

  if (!Array.isArray(initialData)) {
    return []
  }

  return initialData.filter(isSeed).map(seed => ({ ...seed, status: 'unknown', target: '' }))
}

export function clearPersisted(scope: ProfileScope): void {
  writeJson(keyFor(profileScopeKey(scope)), null)
}
