/**
 * The v1 -> v2 bot-metadata migration and the commit protocol that guards it.
 *
 * v1 keyed appearance by bare bot name, which collides the moment two sources
 * expose a same-named profile; v2 keys by `connectionId::profile`. Re-keying is
 * only sound on a topology where every v1 name provably means the LOCAL bot, so
 * migration refuses anything else and leaves v1 in place rather than guessing.
 *
 * The commit is three storage operations that cannot be made atomic (clear the
 * marker, write the snapshot, set the marker), so the marker — not the snapshot
 * — is what makes v2 authoritative. That is what makes a crash mid-write
 * survivable: a markerless v2 is ignored and v1 still loads. A failed commit
 * rolls back to the last committed generation, or clears both keys when there
 * is no generation to roll back to.
 *
 * Ported from the migration half of
 * tests/remote-routing-races.test.mjs, which drove a `vm` copy of plugin.js.
 * `migratedLocalRoutes` is module-private, so where that suite asserted the
 * map's size this one asserts what the map is FOR: no route is adopted and
 * nothing is written.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const { hostMock } = vi.hoisted(() => ({
  hostMock: { agents: vi.fn(), profileRoutes: vi.fn() }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return { atom, host: hostMock, queryClient: undefined, useQuery: vi.fn(), useValue: vi.fn() }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

type Operation = ['remove', string] | ['set', string, unknown]

interface RecordingStorage {
  get: (key: string) => Promise<unknown>
  operations: Operation[]
  remove: (key: string) => Promise<void>
  set: (key: string, value: unknown) => Promise<void>
}

/** A ctx.storage that records every operation and can be told to fail specific
 *  ones — the only way to reach the rollback and crash-window branches. */
function recordingStorage(
  initial: Record<string, unknown>,
  { failRemove = [], failSet = null, failSetTimes = Infinity }: FailureSpec = {}
): RecordingStorage {
  const values = new Map(Object.entries(initial))
  const operations: Operation[] = []
  let failedSets = 0

  return {
    get: async key => values.get(key) ?? null,
    operations,
    remove: async key => {
      operations.push(['remove', key])

      if (failRemove.includes(key)) {
        throw new Error('remove failed')
      }

      values.delete(key)
    },
    set: async (key, value) => {
      operations.push(['set', key, value])

      if (key === failSet && failedSets < failSetTimes) {
        failedSets += 1
        throw new Error('disk full')
      }

      values.set(key, value)
    }
  }
}

interface FailureSpec {
  failRemove?: string[]
  failSet?: null | string
  failSetTimes?: number
}

/** A fresh module graph — the commit chain and the migrated-route map are
 *  module state, so a reload is what "the user restarts the app" means here. */
async function loadData() {
  vi.resetModules()

  return import('./data')
}

/** The one topology migration accepts: a single local source owning the bot. */
function soleLocalTopology(profile: string, targetProfile = profile) {
  hostMock.agents.mockResolvedValue({
    agents: [{ connectionId: 'local', connectionKind: 'local', profile }],
    sources: [{ connectionId: 'local', kind: 'local' }]
  })
  hostMock.profileRoutes.mockResolvedValue([{ connectionId: 'local', mode: 'local', profile, targetProfile }])
}

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.agents.mockResolvedValue({ agents: [], sources: [] })
  hostMock.profileRoutes.mockResolvedValue([])
})

describe('v1 is re-keyed only where the new key is provable', () => {
  it('migrates a sole-local topology and commits the marker after the data', async () => {
    soleLocalTopology('default')

    const storage = recordingStorage({ 'bot-meta': { default: { title: 'Local' } } })
    const { $botMeta, migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(true)

    expect(Object.keys($botMeta.get())).toEqual(['local::default'])
    expect(storage.operations.map(([op, key]) => `${op} ${key}`)).toEqual([
      'remove bot-meta-v2-migrated',
      'set bot-meta-v2',
      'set bot-meta-v2-migrated'
    ])
  })

  it('refuses a multi-source topology rather than project v1 onto a same-named twin', async () => {
    hostMock.agents.mockResolvedValue({
      agents: [
        { connectionId: 'local', connectionKind: 'local', profile: 'default' },
        { connectionId: 'remote-a', connectionKind: 'remote', profile: 'default' }
      ],
      sources: [
        { connectionId: 'local', kind: 'local' },
        { connectionId: 'remote-a', kind: 'remote' }
      ]
    })
    hostMock.profileRoutes.mockResolvedValue([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      { connectionId: 'remote-a', mode: 'remote', profile: 'default', targetProfile: 'remote-default' }
    ])

    const storage = recordingStorage({ 'bot-meta': { default: { title: 'Legacy local' } } })
    const { $botMeta, migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(false)

    // v1 still loads, under its original bare-name key.
    expect($botMeta.get()).toEqual({ default: { title: 'Legacy local' } })
    expect(storage.operations).toEqual([])
  })

  it('refuses the whole batch when even one v1 profile has no local route', async () => {
    // All-or-nothing: adopting routes for the profiles that DO resolve would
    // half-migrate the snapshot, and the unresolved half would keep its bare
    // key forever.
    soleLocalTopology('first', 'backend-first')

    const storage = recordingStorage({
      'bot-meta': { first: { title: 'First' }, missing: { title: 'Missing' } }
    })

    const { $botMeta, migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(false)

    expect($botMeta.get()).toEqual({ first: { title: 'First' }, missing: { title: 'Missing' } })
    expect(storage.operations).toEqual([])
  })
})

describe('the marker, not the snapshot, is what makes v2 authoritative', () => {
  it('hydrates a committed v2 without rewriting it', async () => {
    const v2 = { 'remote-a::default': { title: 'A' }, 'remote-b::default': { title: 'B' } }
    const storage = recordingStorage({ 'bot-meta-v2': v2, 'bot-meta-v2-migrated': true })
    const { $botMeta, migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(true)

    expect($botMeta.get()).toEqual(v2)
    expect(storage.operations).toEqual([])
  })

  it('ignores a markerless v2 instead of adopting it implicitly', async () => {
    const storage = recordingStorage({ 'bot-meta-v2': { 'remote-a::default': { title: 'Uncommitted' } } })
    const { $botMeta, migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(false)

    expect($botMeta.get()).toEqual({})
    expect(storage.operations).toEqual([])
  })

  it('reloads v1 when a crash left a markerless v2 behind', async () => {
    // The crash window is between the snapshot write and the marker write.
    // Whatever v2 holds there is a partial generation; v1 is the last good one.
    const storage = recordingStorage({
      'bot-meta': { first: { title: 'Rollback' } },
      'bot-meta-v2': { 'local::first': { title: 'Crash window' } }
    })

    const { $botMeta, migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(false)

    expect($botMeta.get()).toEqual({ first: { title: 'Rollback' } })
  })
})

describe('a failed commit falls back to the last committed generation', () => {
  it('restores the previous snapshot when the marker write fails', async () => {
    const committed = { 'remote-a::default': { title: 'B' } }

    const storage = recordingStorage(
      { 'bot-meta-v2': committed, 'bot-meta-v2-migrated': true },
      { failSet: 'bot-meta-v2-migrated', failSetTimes: 1 }
    )

    const { commitBotMetaV2 } = await loadData()

    await expect(commitBotMetaV2(storage, { 'remote-a::default': { title: 'D' } })).rejects.toThrow('disk full')

    // Reload: the successor never became authoritative, so the generation the
    // marker last vouched for is what comes back.
    const reloaded = await loadData()

    await expect(reloaded.migrateBotMeta(storage)).resolves.toBe(true)
    expect(reloaded.$botMeta.get()).toEqual(committed)
  })

  it('clears both keys when the failed commit had no committed generation behind it', async () => {
    // First-ever migration: rolling "back" to a previous v2 is not an option,
    // so both keys must go, leaving v1 authoritative again.
    soleLocalTopology('first', 'backend-first')

    const storage = recordingStorage({ 'bot-meta': { first: { title: 'First' } } }, { failSet: 'bot-meta-v2-migrated' })

    const { $botMeta, migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(false)

    expect($botMeta.get()).toEqual({ first: { title: 'First' } })
    expect(storage.operations.map(([op, key]) => `${op} ${key}`)).toEqual([
      'remove bot-meta-v2-migrated',
      'set bot-meta-v2',
      'set bot-meta-v2-migrated',
      'remove bot-meta-v2-migrated',
      'remove bot-meta-v2'
    ])

    const reloaded = await loadData()

    await expect(reloaded.migrateBotMeta(storage)).resolves.toBe(false)
    expect(reloaded.$botMeta.get()).toEqual({ first: { title: 'First' } })
  })

  it('still reloads v1 when even the rollback cleanup fails', async () => {
    // Worst case: marker write fails AND the v2 cleanup fails, so a markerless
    // v2 is left on disk. The marker rule is what saves it — v1 still wins.
    soleLocalTopology('first', 'backend-first')

    const storage = recordingStorage(
      { 'bot-meta': { first: { title: 'Rollback' } } },
      { failRemove: ['bot-meta-v2'], failSet: 'bot-meta-v2-migrated' }
    )

    const { migrateBotMeta } = await loadData()

    await expect(migrateBotMeta(storage)).resolves.toBe(false)

    const reloaded = await loadData()

    await expect(reloaded.migrateBotMeta(storage)).resolves.toBe(false)
    expect(reloaded.$botMeta.get()).toEqual({ first: { title: 'Rollback' } })
  })
})
