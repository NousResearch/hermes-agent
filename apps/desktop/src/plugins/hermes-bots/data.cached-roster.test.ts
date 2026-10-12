/**
 * `cachedUnionRoster` is the imperative roster read — the @mention popover
 * must answer per keystroke and the composer middleware runs on submit, so
 * neither can wait on the hook.
 *
 * `useRoster` keys its query on `[...ROSTER_KEY, connectionId]`, one entry per
 * connection the window has been on. Reading it back with the BARE key is an
 * exact-key match in TanStack Query and therefore matches NOTHING — the
 * regression where completions offered no handles and remote `@name-device`
 * mentions passed through unresolved. The read has to fall back to a prefix
 * match over the key family, newest snapshot wins.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const { cache, connection } = vi.hoisted(() => ({
  cache: new Map<string, { key: unknown[]; value: unknown }>(),
  connection: { id: 'local' }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')
  const keyOf = (key: unknown[]) => JSON.stringify(key)

  return {
    atom,
    host: { state: { connectionId: { get: () => connection.id }, profile: { get: () => 'default' } } },
    queryClient: {
      getQueriesData: ({ queryKey }: { queryKey: unknown[] }) =>
        [...cache.values()]
          .filter(entry => queryKey.every((part, index) => entry.key[index] === part))
          .map(entry => [entry.key, entry.value]),
      getQueryData: (key: unknown[]) => cache.get(keyOf(key))?.value,
      invalidateQueries: vi.fn(),
      setQueryData: (key: unknown[], value: unknown) => cache.set(keyOf(key), { key, value })
    },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

const seed = (key: unknown[], value: unknown) => cache.set(JSON.stringify(key), { key, value })

beforeEach(() => {
  cache.clear()
  connection.id = 'local'
})

describe('cachedUnionRoster', () => {
  it('reads the entry useRoster wrote under the connection-suffixed key', async () => {
    const { cachedUnionRoster } = await import('./data')

    seed(['hermes-bots', 'roster', 'local'], { profiles: [{ name: 'default' }] })

    expect(cachedUnionRoster()?.profiles).toHaveLength(1)
    // The bare key is what the broken read used — it must still miss, or this
    // test would pass for the wrong reason.
    expect(cache.has(JSON.stringify(['hermes-bots', 'roster']))).toBe(false)
  })

  it('falls back to another connection’s entry when the window has moved', async () => {
    const { cachedUnionRoster } = await import('./data')

    seed(['hermes-bots', 'roster', 'vera'], { profiles: [{ connectionId: 'vera', name: 'default' }] })
    connection.id = 'local'

    expect(cachedUnionRoster()?.profiles?.[0]).toMatchObject({ connectionId: 'vera' })
  })

  it('prefers the freshest snapshot among several cached connections', async () => {
    const { cachedUnionRoster } = await import('./data')

    seed(['hermes-bots', 'roster', 'old'], { fetchedAt: 1_000, profiles: [{ name: 'stale' }] })
    seed(['hermes-bots', 'roster', 'new'], { fetchedAt: 9_000, profiles: [{ name: 'fresh' }] })
    connection.id = 'neither'

    expect(cachedUnionRoster()?.profiles?.[0]).toMatchObject({ name: 'fresh' })
  })

  it('reports nothing rather than throwing on a cold cache', async () => {
    const { cachedUnionRoster } = await import('./data')

    expect(cachedUnionRoster()).toBeNull()
  })
})
