import { type ProfileScope, profileScopeKey } from '@/rabbit'
import { queryClient } from '@/lib/query-client'
import { readJson, writeJson } from '@/lib/storage'

import { MCP_CATALOG_KEY } from '../../mcp/mcp-status'
import type { LocalServerInput } from '../types'

export type PersistedRead = 'bundled' | 'servers'

const STORAGE_PREFIX = 'rabbit.connectors.v4.'

const PERSIST_MAX_BYTES = 256 * 1024

interface PersistedEntry {
  at: number
  data: unknown
}

type PersistedValue = PersistedEntry['data']

interface PersistedBlob {
  bundled?: PersistedEntry
  servers?: PersistedEntry
}

const keyFor = (scopeKey: string) => `${STORAGE_PREFIX}${scopeKey}`

const SLOTS = {
  bundled: 'bundled',
  servers: 'servers'
} satisfies Record<PersistedRead, keyof PersistedBlob>

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

  return blob
}

function entryOf(blob: PersistedBlob, read: PersistedRead): PersistedEntry | undefined {
  return blob[SLOTS[read]]
}

export interface QuerySeed<T> {
  initialData?: T
  initialDataUpdatedAt?: number
}

export function seedOptions<T>(scopeKey: ProfileScope, read: PersistedRead): QuerySeed<T> {
  const blob = readBlob(profileScopeKey(scopeKey))
  const entry = blob ? entryOf(blob, read) : undefined

  if (!entry || !Number.isFinite(entry.at) || entry.data === undefined || entry.data === null) {
    return {}
  }

  // SAFETY: the caller names the read whose answer it stored, so the entry holds that read's own result.
  return { initialData: entry.data as T, initialDataUpdatedAt: entry.at }
}

function store(scopeKey: string, read: PersistedRead, entry: PersistedEntry): void {
  if (size(entry) > PERSIST_MAX_BYTES) {
    return
  }

  const blob = readBlob(scopeKey) ?? {}
  blob[SLOTS[read]] = entry

  writeJson(keyFor(scopeKey), blob)
}

interface ReadTarget {
  read: PersistedRead
  scopeKey: string
}

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
      store(write.scopeKey, write.read, write.entry)
      written.set(key, write.entry.at)
    }

    pending.clear()
  }

  const stopCache = queryClient.getQueryCache().subscribe(event => {
    if (event.type !== 'updated' || event.action.type !== 'success') {
      return
    }

    const [root, scopeKey] = event.query.queryKey

    if (root !== MCP_CATALOG_KEY[0] || typeof scopeKey !== 'string') {
      return
    }

    const { data, dataUpdatedAt } = event.query.state

    if (data === undefined) {
      return
    }

    const key = `${scopeKey}bundled`

    if (written.get(key) === dataUpdatedAt) {
      return
    }

    pending.set(key, { read: 'bundled', scopeKey, entry: { at: dataUpdatedAt, data } })
    timer ??= setTimeout(flush, WRITE_DELAY_MS)
  })

  return () => {
    stopCache()

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

  store(profileScopeKey(scope), 'servers', { at: Date.now(), data: seeds })
}

export function seedLocalServers(scope: ProfileScope): LocalServerInput[] {
  const { initialData } = seedOptions<unknown>(scope, 'servers')

  if (!Array.isArray(initialData)) {
    return []
  }

  return initialData.filter(isSeed).map(seed => ({ ...seed, status: 'unknown', target: '' }))
}
