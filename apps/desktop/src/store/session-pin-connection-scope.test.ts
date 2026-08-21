import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { connectionScopeSuffix } from '@/lib/connection-scoped'
import { readKey, storedStringArray, writeKey } from '@/lib/storage'
import type { SessionInfo } from '@/types/hermes'

const patch = vi.fn<(id: string, pinned: boolean, profile?: null | string) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

vi.mock('@/hermes', () => ({
  setApiRequestProfile: () => {},
  setSessionPinnedRemote: (id: string, pinned: boolean, profile?: null | string) => patch(id, pinned, profile)
}))

import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $sessions, setConnection } from '@/store/session'

import { resetSessionPinMirror, watchSessionPins } from './session-pin-sync'

const PIN_KEY = 'hermes.desktop.pinnedSessions'

const remote = (profile: string, baseUrl = 'https://gw.example:8443'): HermesConnection =>
  ({
    baseUrl,
    mode: 'remote',
    profile,
    token: 't',
    wsUrl: 'ws://x'
  }) as unknown as HermesConnection

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

const flush = () => Promise.resolve()

beforeAll(() => {
  ;(globalThis as { window?: unknown }).window ??= {}
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}
  watchSessionPins()
})

beforeEach(() => {
  window.localStorage.clear()
  setConnection(remote('default'))
  $sessions.set([])
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
  patch.mockClear()
})

afterEach(() => {
  $sessions.set([])
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
})

describe('desktop pin list is connection-scoped, not profile-scoped', () => {
  it('keeps the pin storage key stable across a profile switch', () => {
    setConnection(remote('default'))
    pinSession('s1')

    const gatewayKey = `${PIN_KEY}${connectionScopeSuffix(remote('default'), false)}`

    expect(readKey(gatewayKey)).toBe(JSON.stringify(['s1']))

    setConnection(remote('k9'))
    expect($pinnedSessionIds.get()).toEqual(['s1'])
    expect(readKey(gatewayKey)).toBe(JSON.stringify(['s1']))
    expect(readKey(`${PIN_KEY}${connectionScopeSuffix(remote('k9'))}`)).toBeNull()
  })

  it('still isolates pin sets between two different remote gateways', () => {
    setConnection(remote('default', 'https://gw-a.example'))
    pinSession('a-1')

    setConnection(remote('default', 'https://gw-b.example'))
    expect($pinnedSessionIds.get()).toEqual([])
    pinSession('b-1')

    setConnection(remote('k9', 'https://gw-a.example'))
    expect($pinnedSessionIds.get()).toEqual(['a-1'])

    setConnection(remote('default', 'https://gw-b.example'))
    expect($pinnedSessionIds.get()).toEqual(['b-1'])
  })

  it('lets an unpin survive a profile rescope instead of flushing pin=true', async () => {
    $sessions.set([row('s1', { pinned: false, profile: 'k9' })])

    setConnection(remote('default'))
    pinSession('s1')
    await flush()

    setConnection(remote('k9'))
    expect($pinnedSessionIds.get()).toEqual(['s1'])

    unpinSession('s1')
    await flush()
    patch.mockClear()

    setConnection(remote('default'))
    await flush()

    expect($pinnedSessionIds.get()).not.toContain('s1')
    expect(patch).not.toHaveBeenCalledWith('s1', true, expect.anything())
    expect(patch).not.toHaveBeenCalledWith('s1', true, 'k9')
  })
})

describe('upgrade from per-profile pin keys is server-authoritative', () => {
  const gateway = 'https://gw.example:8443'
  const gatewayKey = () => `${PIN_KEY}${connectionScopeSuffix(remote('default', gateway), false)}`
  const legacyKey = (profile: string) => `${PIN_KEY}${connectionScopeSuffix(remote(profile, gateway))}`

  const seedStalePerProfilePins = () => {
    // First launch after the gateway-wide key: old profile fragments still
    // sit in localStorage, the new key is absent, and the in-memory set is
    // empty. Those fragments caused #90021 — they must not be unioned in.
    window.localStorage.clear()
    writeKey(legacyKey('default'), JSON.stringify(['s1']))
    writeKey(legacyKey('k9'), JSON.stringify(['s1']))
    setConnection(remote('default', gateway))
    $sessions.set([])
    resetSessionPinMirror()
    patch.mockClear()
  }

  it('does not resurrect a stale per-profile pin when the server says unpinned', async () => {
    seedStalePerProfilePins()
    expect(readKey(gatewayKey())).toBeNull()

    $sessions.set([row('s1', { pinned: false, profile: 'k9' })])
    await flush()

    setConnection(remote('k9', gateway))
    await flush()
    setConnection(remote('default', gateway))
    await flush()

    expect($pinnedSessionIds.get()).not.toContain('s1')
    expect(storedStringArray(gatewayKey())).not.toContain('s1')
    expect(patch).not.toHaveBeenCalledWith('s1', true, expect.anything())
    expect(patch).not.toHaveBeenCalledWith('s1', true, 'k9')
    expect(readKey(legacyKey('default'))).toBe(JSON.stringify(['s1']))
    expect(readKey(legacyKey('k9'))).toBe(JSON.stringify(['s1']))
  })

  it('repopulates the gateway-wide cache from a durable server pin without echoing PATCH', async () => {
    seedStalePerProfilePins()
    expect(readKey(gatewayKey())).toBeNull()

    $sessions.set([row('s1', { pinned: true, profile: 'k9' })])
    await flush()

    expect($pinnedSessionIds.get()).toContain('s1')
    expect(storedStringArray(gatewayKey())).toContain('s1')
    expect(patch).not.toHaveBeenCalledWith('s1', true, expect.anything())
    expect(readKey(legacyKey('default'))).toBe(JSON.stringify(['s1']))
    expect(readKey(legacyKey('k9'))).toBe(JSON.stringify(['s1']))
  })
})
