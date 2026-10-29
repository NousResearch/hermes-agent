import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  coldRegistryConnectionIds,
  FEDERATED_SESSION_INDEX_VERSION,
  type FederatedSessionIndex,
  FederatedSessionIndexStore,
  filterIndexedRows,
  indexedRowsForColdConnections,
  indexedSessionRows,
  MAX_OWNER_IDLE_MS,
  mergeSessionIndexRows,
  parseSessionIndex,
  pruneSessionIndex,
  retainIndexedConnections,
  SESSION_INDEX_FILE_NAME
} from './federated-session-index'

const NOW = 1_900_000_000_000 // fixed epoch ms for deterministic pruning

function row(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    id: 's1',
    profile: 'default',
    connection_id: 'homelab',
    title: 'Fix inbox',
    last_active: 1_899_000_000,
    started_at: 1_899_000_000,
    ...overrides
  }
}

function emptyIndex() {
  return { version: FEDERATED_SESSION_INDEX_VERSION, owners: {} }
}

describe('parseSessionIndex', () => {
  it('accepts a v1 payload with per-owner rows', () => {
    const parsed = parseSessionIndex({
      version: 1,
      owners: { 'homelab::default': [row()] }
    })

    expect(parsed?.owners['homelab::default']).toHaveLength(1)
  })

  it('rejects an unknown version (cache treats as absent and rebuilds)', () => {
    expect(parseSessionIndex({ version: 2, owners: {} })).toBeNull()
    expect(parseSessionIndex({ owners: {} })).toBeNull()
    expect(parseSessionIndex(null)).toBeNull()
  })

  it('drops malformed owner entries and rows without owner tags', () => {
    const parsed = parseSessionIndex({
      version: 1,
      owners: {
        ok: [row()],
        notAnArray: { nope: true },
        bad: [{ id: 'x' /* no connection_id */ }, row()]
      }
    })

    expect(Object.keys(parsed?.owners ?? {})).toEqual(['ok', 'bad'])
    expect(parsed?.owners.bad).toHaveLength(1)
  })
})

describe('mergeSessionIndexRows', () => {
  it('groups one shared-host page by each row own profile tag', () => {
    const next = mergeSessionIndexRows(
      emptyIndex(),
      'homelab',
      { sessions: [row(), row({ id: 's2', profile: 'omer' })] },
      NOW
    )

    expect(Object.keys(next.owners).sort()).toEqual(['homelab::default', 'homelab::omer'])
  })

  it('ignores rows owned by a different connection', () => {
    const next = mergeSessionIndexRows(emptyIndex(), 'homelab', { sessions: [row({ connection_id: 'other' })] }, NOW)

    expect(next.owners).toEqual({})
  })

  it('a fresh read REPLACES the owner profile rows (delete on host clears the cache)', () => {
    let index = mergeSessionIndexRows(emptyIndex(), 'homelab', { sessions: [row()] }, NOW)
    index = mergeSessionIndexRows(index, 'homelab', { sessions: [] }, NOW)

    expect(index.owners).toEqual({})
  })

  it('clears a profile of the connection that a fresh read no longer mentions', () => {
    let index = mergeSessionIndexRows(
      emptyIndex(),
      'homelab',
      { sessions: [row(), row({ id: 's2', profile: 'omer' })] },
      NOW
    )

    index = mergeSessionIndexRows(index, 'homelab', { sessions: [row()] }, NOW)

    expect(Object.keys(index.owners)).toEqual(['homelab::default'])
  })

  it('caps an owner at MAX_ROWS_PER_OWNER, newest first', () => {
    // Recencies must sit inside the prune window (14d before NOW) so the cap,
    // not the idle prune, is what the assertion observes.
    const many = Array.from({ length: 250 }, (_, i) =>
      row({ id: `s${i}`, last_active: 1_899_000_000 + i, started_at: 1_899_000_000 + i })
    )

    const next = mergeSessionIndexRows(emptyIndex(), 'homelab', { sessions: many }, NOW)

    expect(next.owners['homelab::default']).toHaveLength(200)
    expect(next.owners['homelab::default'][0].id).toBe('s249')
  })
})

describe('pruneSessionIndex', () => {
  it('drops owners idle beyond MAX_OWNER_IDLE_MS and keeps recent ones', () => {
    const recent = row({ id: 'recent' })
    const stale = row({ id: 'stale', connection_id: 'oldbox', last_active: 1, started_at: 1 })

    const index = pruneSessionIndex(
      {
        version: FEDERATED_SESSION_INDEX_VERSION,
        owners: { 'homelab::default': [recent], 'oldbox::default': [stale] }
      },
      NOW
    )

    expect(Object.keys(index.owners)).toEqual(['homelab::default'])
    expect(MAX_OWNER_IDLE_MS).toBeGreaterThan(0)
  })
})

describe('indexedSessionRows', () => {
  it('flattens every owner, newest-first', () => {
    const index = {
      version: FEDERATED_SESSION_INDEX_VERSION,
      owners: {
        'a::default': [row({ id: 'old-a', last_active: 5, started_at: 5 })],
        'b::default': [row({ id: 'new-b', connection_id: 'b', last_active: 9, started_at: 9 })]
      }
    }

    expect(indexedSessionRows(index).map(r => r.id)).toEqual(['new-b', 'old-a'])
  })
})

describe('retainIndexedConnections', () => {
  it('drops owners of connections that left the registry and keeps the rest', () => {
    const index: FederatedSessionIndex = {
      version: FEDERATED_SESSION_INDEX_VERSION,
      owners: {
        'homelab::default': [row()],
        'homelab::omer': [row({ profile: 'omer' })],
        'gone::default': [row({ connection_id: 'gone' })]
      }
    }

    const next = retainIndexedConnections(index, new Set(['homelab']))

    expect(Object.keys(next.owners).sort()).toEqual(['homelab::default', 'homelab::omer'])
  })

  it('returns the same index object when nothing is dropped (no needless rewrite)', () => {
    const index: FederatedSessionIndex = {
      version: FEDERATED_SESSION_INDEX_VERSION,
      owners: { 'homelab::default': [row()] }
    }

    expect(retainIndexedConnections(index, new Set(['homelab']))).toBe(index)
  })
})

describe('filterIndexedRows', () => {
  const rows = [
    row({ id: 'chat', source: 'cli', message_count: 4 }),
    row({ id: 'cron-run', source: 'cron', message_count: 2 }),
    row({ id: 'telegram', source: 'telegram', message_count: 1 }),
    row({ id: 'empty', source: 'cli', message_count: 0 })
  ]

  const ids = (query: string) => filterIndexedRows(rows, new URLSearchParams(query)).map(r => r.id)

  it('passes every row through an unfiltered request', () => {
    expect(ids('')).toEqual(['chat', 'cron-run', 'telegram', 'empty'])
  })

  it('serves the cron slice only cron rows', () => {
    expect(ids('source=cron')).toEqual(['cron-run'])
  })

  it('keeps cron and messaging rows out of a recents slice that excludes them', () => {
    expect(ids('exclude_sources=cron,telegram&min_messages=1')).toEqual(['chat'])
  })

  it('honours an explicit source allow-list', () => {
    expect(ids('sources=telegram,cron')).toEqual(['cron-run', 'telegram'])
  })

  it('applies min_messages', () => {
    expect(ids('min_messages=3')).toEqual(['chat'])
  })

  it('never serves an archived-only request: the index holds non-archived rows only', () => {
    expect(ids('archived=only')).toEqual([])
  })
})

describe('coldRegistryConnectionIds', () => {
  const connections = [
    { id: 'local', kind: 'local' },
    { id: 'homelab', kind: 'ssh' },
    { id: 'cloud', kind: 'cloud' },
    { id: 'office', kind: 'remote' }
  ]

  it('names the non-local, non-primary connections with no readable backend', () => {
    expect([...coldRegistryConnectionIds(connections, 'local', new Set(['cloud']))].sort()).toEqual([
      'homelab',
      'office'
    ])
  })

  it('never treats the registry primary as cold: its rows ride the primary aggregate', () => {
    expect([...coldRegistryConnectionIds(connections, 'homelab', new Set())].sort()).toEqual(['cloud', 'office'])
  })

  it('is empty when every remote connection is readable', () => {
    expect(coldRegistryConnectionIds(connections, 'local', new Set(['homelab', 'cloud', 'office'])).size).toBe(0)
  })
})

describe('indexedRowsForColdConnections', () => {
  const index: FederatedSessionIndex = {
    version: FEDERATED_SESSION_INDEX_VERSION,
    owners: {
      'homelab::default': [row({ id: 'h1', source: 'cli', message_count: 3 })],
      'cloud::default': [row({ id: 'c1', connection_id: 'cloud', source: 'cli', message_count: 3 })]
    }
  }

  it('returns only the cold connections rows', () => {
    const rows = indexedRowsForColdConnections(index, new Set(['homelab']), new URLSearchParams())

    expect(rows.map(r => r.id)).toEqual(['h1'])
  })

  it('returns nothing when no connection is cold (a readable host is never backfilled)', () => {
    expect(indexedRowsForColdConnections(index, new Set(), new URLSearchParams())).toEqual([])
  })

  it('narrows by the request filters', () => {
    expect(indexedRowsForColdConnections(index, new Set(['homelab']), new URLSearchParams('source=cron'))).toEqual([])
  })
})

describe('FederatedSessionIndexStore', () => {
  let dir: string
  let store: FederatedSessionIndexStore

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'hermes-session-index-'))
    store = new FederatedSessionIndexStore(dir)
  })

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true })
  })

  it('starts empty and rebuilds from an absent file', () => {
    expect(store.rows()).toEqual([])
  })

  it('persists across store instances (tmp+rename atomic write)', () => {
    store.remember('homelab', { sessions: [row()] })
    const onDisk = JSON.parse(readFileSync(join(dir, SESSION_INDEX_FILE_NAME), 'utf8'))
    const reopened = new FederatedSessionIndexStore(dir)

    expect(onDisk.version).toBe(1)
    expect(reopened.rows().map(r => r.id)).toEqual(['s1'])
  })

  it('treats a corrupt file as absent (cache rebuilds)', () => {
    writeFileSync(join(dir, SESSION_INDEX_FILE_NAME), '{not json', 'utf8')
    const reopened = new FederatedSessionIndexStore(dir)

    expect(reopened.rows()).toEqual([])
  })

  it('clear() removes the file and empties the cache', () => {
    store.remember('homelab', { sessions: [row()] })
    store.clear()

    expect(store.rows()).toEqual([])
  })

  it('retainConnections() forgets a removed connection on disk too', () => {
    store.remember('homelab', { sessions: [row()] })
    store.remember('gone', { sessions: [row({ id: 'g1', connection_id: 'gone' })] })
    store.retainConnections(new Set(['homelab']))

    const reopened = new FederatedSessionIndexStore(dir)

    expect(reopened.rows().map(r => r.id)).toEqual(['s1'])
  })
})
