import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// The pin store's scope state (`activePinnedScope`, persistence binding) is
// module-level, so every test boots a FRESH layout module against a clean
// localStorage — that's what lets a test simulate "a fresh window booting
// against existing stored pins".

const LEGACY_KEY = 'hermes.desktop.pinnedSessions'

const MIGRATED_KEY = 'hermes.desktop.pinnedSessions.scope-migrated'

const localConnection = (): HermesConnection =>
  ({ baseUrl: 'http://127.0.0.1:41001', mode: 'local', token: 't', wsUrl: 'ws://x' }) as HermesConnection

const remoteConnection = (baseUrl: string, profile = 'default'): HermesConnection =>
  ({ baseUrl, mode: 'remote', profile, token: 't', wsUrl: 'ws://x' }) as HermesConnection

async function freshLayout() {
  vi.resetModules()

  return import('@/store/layout')
}

beforeEach(() => {
  localStorage.clear()
})

describe('connectionScopeId', () => {
  it('is local for a local-mode connection', async () => {
    const { connectionScopeId } = await freshLayout()

    expect(connectionScopeId(localConnection())).toBe('local')
  })

  it('is local for a null connection', async () => {
    const { connectionScopeId } = await freshLayout()

    expect(connectionScopeId(null)).toBe('local')
  })

  it('is remote.<baseUrl>.<profile> for a remote connection', async () => {
    const { connectionScopeId } = await freshLayout()

    expect(connectionScopeId(remoteConnection('https://gw.example.com'))).toBe(
      'remote.https%3A%2F%2Fgw.example.com.default'
    )
  })

  it('includes the profile so two profiles on one gateway stay apart', async () => {
    const { connectionScopeId } = await freshLayout()

    expect(connectionScopeId(remoteConnection('https://gw', 'work'))).toBe('remote.https%3A%2F%2Fgw.work')
  })
})

describe('applyPinnedSessionScope', () => {
  it('seeds the store from the scope key and persists pins to it', async () => {
    localStorage.setItem('hermes.desktop.pinnedSessions.local', JSON.stringify(['a', 'b']))

    const { $pinnedSessionIds, applyPinnedSessionScope, pinSession } = await freshLayout()

    applyPinnedSessionScope(localConnection())
    expect($pinnedSessionIds.get()).toEqual(['a', 'b'])

    pinSession('c')
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.local')).toBe(JSON.stringify(['a', 'b', 'c']))
    // Pins must never write back to the pre-scope legacy key.
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
  })

  it('isolates pin sets between two gateways', async () => {
    const { $pinnedSessionIds, applyPinnedSessionScope, pinSession, unpinSession } = await freshLayout()

    // Window A on gateway A pins two sessions.
    applyPinnedSessionScope(remoteConnection('https://gw-a'))
    pinSession('a1')
    pinSession('a2')
    expect($pinnedSessionIds.get()).toEqual(['a1', 'a2'])

    // Window B on gateway B must NOT see A's pins.
    applyPinnedSessionScope(remoteConnection('https://gw-b'))
    expect($pinnedSessionIds.get()).toEqual([])

    // B pins its own; A's pins are untouched in storage.
    pinSession('b1')
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-b.default')).toBe(
      JSON.stringify(['b1'])
    )
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-a.default')).toBe(
      JSON.stringify(['a1', 'a2'])
    )

    // Back on gateway A, the local set is A's again.
    applyPinnedSessionScope(remoteConnection('https://gw-a'))
    expect($pinnedSessionIds.get()).toEqual(['a1', 'a2'])

    // And unpinning on A only affects A's key.
    unpinSession('a1')
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-a.default')).toBe(
      JSON.stringify(['a2'])
    )
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-b.default')).toBe(
      JSON.stringify(['b1'])
    )
  })

  it('keeps the pins when the same gateway re-applies its scope (reconnect)', async () => {
    const { $pinnedSessionIds, applyPinnedSessionScope, pinSession } = await freshLayout()

    applyPinnedSessionScope(remoteConnection('https://gw'))
    pinSession('p1')

    // A reconnect re-publishes the same connection: scope unchanged, no wipe,
    // no reseed over the user's in-memory set.
    applyPinnedSessionScope(remoteConnection('https://gw'))
    expect($pinnedSessionIds.get()).toEqual(['p1'])
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw.default')).toBe(
      JSON.stringify(['p1'])
    )
  })

  it('is a no-op for a null connection (transient request failure)', async () => {
    const { $pinnedSessionIds, applyPinnedSessionScope, pinSession } = await freshLayout()

    applyPinnedSessionScope(remoteConnection('https://gw'))
    pinSession('p1')

    applyPinnedSessionScope(null)
    expect($pinnedSessionIds.get()).toEqual(['p1'])
  })

  it('distinguishes local from remote instead of sharing one key', async () => {
    const { $pinnedSessionIds, applyPinnedSessionScope, pinSession } = await freshLayout()

    applyPinnedSessionScope(localConnection())
    pinSession('local-pin')

    applyPinnedSessionScope(remoteConnection('https://vps'))
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('restores a window booting fresh against its own stored scope', async () => {
    localStorage.setItem(
      'hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-a.default',
      JSON.stringify(['x', 'y'])
    )
    localStorage.setItem(
      'hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-b.default',
      JSON.stringify(['z'])
    )

    // Fresh window on gateway A → only A's pins.
    const first = await freshLayout()
    first.applyPinnedSessionScope(remoteConnection('https://gw-a'))
    expect(first.$pinnedSessionIds.get()).toEqual(['x', 'y'])

    // Fresh window on gateway B → only B's pins.
    const second = await freshLayout()
    second.applyPinnedSessionScope(remoteConnection('https://gw-b'))
    expect(second.$pinnedSessionIds.get()).toEqual(['z'])
  })
})

describe('pre-scope migration', () => {
  it('lets the first window claim the legacy pin set, then isolates every other gateway', async () => {
    localStorage.setItem(LEGACY_KEY, JSON.stringify(['legacy-1', 'legacy-2']))

    const { $pinnedSessionIds, applyPinnedSessionScope, pinSession } = await freshLayout()

    // First window (gateway A) claims the pre-scope pins.
    applyPinnedSessionScope(remoteConnection('https://gw-a'))
    expect($pinnedSessionIds.get()).toEqual(['legacy-1', 'legacy-2'])
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-a.default')).toBe(
      JSON.stringify(['legacy-1', 'legacy-2'])
    )
    expect(localStorage.getItem(MIGRATED_KEY)).toBe('true')

    // A second gateway must NOT inherit the legacy set.
    applyPinnedSessionScope(remoteConnection('https://gw-b'))
    expect($pinnedSessionIds.get()).toEqual([])

    // Pins made after migration stay in their own scope.
    pinSession('fresh')
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-b.default')).toBe(
      JSON.stringify(['fresh'])
    )
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.remote.https%3A%2F%2Fgw-a.default')).toBe(
      JSON.stringify(['legacy-1', 'legacy-2'])
    )
  })

  it('does not claim the legacy set once the migration marker exists', async () => {
    localStorage.setItem(LEGACY_KEY, JSON.stringify(['legacy-1']))
    localStorage.setItem(MIGRATED_KEY, 'true')

    const { $pinnedSessionIds, applyPinnedSessionScope } = await freshLayout()

    applyPinnedSessionScope(remoteConnection('https://gw'))
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('migrates a local-first install into the local scope key', async () => {
    localStorage.setItem(LEGACY_KEY, JSON.stringify(['legacy-1']))

    const { $pinnedSessionIds, applyPinnedSessionScope } = await freshLayout()

    applyPinnedSessionScope(localConnection())
    expect($pinnedSessionIds.get()).toEqual(['legacy-1'])
    expect(localStorage.getItem('hermes.desktop.pinnedSessions.local')).toBe(JSON.stringify(['legacy-1']))
    // A later remote window on the same machine starts empty, not with local pins.
    applyPinnedSessionScope(remoteConnection('https://vps'))
    expect($pinnedSessionIds.get()).toEqual([])
  })
})
