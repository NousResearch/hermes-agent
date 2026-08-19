import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { connectionScopeSuffix } from '@/lib/connection-scoped'
import { readKey } from '@/lib/storage'
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
