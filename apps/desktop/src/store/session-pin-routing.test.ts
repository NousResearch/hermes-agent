import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $pinnedSessionIds } from '@/store/layout'
import { $sessions, setConnection } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { resetSessionPinMirror, watchSessionPins } from './session-pin-sync'

const api = vi.fn(async (_request: unknown) => ({ ok: true }) as never)

const connection = {
  baseUrl: 'https://gateway.example.test',
  mode: 'remote',
  profile: 'default',
  remoteKind: 'url'
} as HermesConnection

const row = (id: string): SessionInfo =>
  ({ id, message_count: 1, profile: 'work', source: 'cli', started_at: 0, title: id }) as SessionInfo

const flush = () => Promise.resolve()

let stopWatching = () => {}

beforeAll(() => {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { api }
  stopWatching = watchSessionPins()
})

beforeEach(() => {
  setConnection(null)
  window.localStorage.clear()
  $sessions.set([])
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
  api.mockClear()
})

afterEach(() => {
  setConnection(null)
  $sessions.set([])
  $pinnedSessionIds.set([])
  window.localStorage.clear()
})

afterAll(() => {
  stopWatching()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('session pin bridge routing', () => {
  it('carries the session owner through pin sync into the Desktop API request', async () => {
    // The active connection and session owner deliberately differ: Electron
    // must route the mutation by the row's owner, not the foreground profile.
    setConnection(connection)
    $sessions.set([row('shared-session')])
    $pinnedSessionIds.set(['shared-session'])
    await flush()

    expect(api).toHaveBeenCalledWith({
      body: { pinned: true },
      method: 'PATCH',
      path: '/api/sessions/shared-session',
      profile: 'work'
    })
  })
})
