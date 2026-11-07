import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type * as HermesModule from '@/hermes'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

// The pin key is scoped to the window's connection, but its CONTENT is not:
// `pullRemotePins` adopts every pinned row the sidebar shows, and the
// all-profiles view splices in other connections' rows (#146912). So the local
// key can hold ids that only a remote gateway owns. This suite pins down what
// the prune does with them: it probes them on the selected connection only, it
// unpins them from THIS key when every profile here says 404, it never PATCHes
// the unpin to any gateway, and it leaves the other connection's key alone.

const getSessionMock =
  vi.fn<(id: string, profile?: null | string | { connectionId: string; profile: string }) => Promise<SessionInfo>>()

const getProfilesMock = vi.fn<() => Promise<{ profiles: ProfileInfo[] }>>()

const patch = vi.fn<(id: string, pinned: boolean, profile?: null | string) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

const logLine = vi.fn<(line: string) => void>()

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesModule>()

  return {
    ...actual,
    getProfiles: () => getProfilesMock(),
    getSession: (id: string, profile?: null | string | { connectionId: string; profile: string }) =>
      getSessionMock(id, profile),
    setSessionPinnedRemote: (id: string, pinned: boolean, profile?: null | string) => patch(id, pinned, profile)
  }
})

import { $pinnedSessionIds } from '@/store/layout'
import { $activeGatewayProfile, $profiles, refreshProfiles } from '@/store/profile'
import { $sessions, _resetSessionOwnerHintsForTests, setConnection, setSessionOwnerHint } from '@/store/session'
import { resetSessionPinMirror, watchSessionPins } from '@/store/session-pin-sync'

import { __runDeadSessionPrunePass, resetDeadSessionPrune } from './dead-session-prune'

const REMOTE_BASE_URL = 'https://a.example'
// `connectionScopeSuffix` for a remote connection's gateway-wide (pins) key.
const REMOTE_PIN_KEY = `hermes.desktop.pinnedSessions.remote.${encodeURIComponent(REMOTE_BASE_URL)}`

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

const profile = (name: string): ProfileInfo =>
  ({
    has_env: false,
    is_default: name === 'default',
    model: null,
    name,
    path: name,
    provider: null,
    skill_count: 0
  }) as ProfileInfo

const notFound = () => new Error('404: {"detail":"Session not found"}')

const flush = () => Promise.resolve()

beforeAll(() => {
  ;(globalThis as { window?: unknown }).window ??= {}
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { logLine }
  // The pin mirror's listeners are module-global; attach them once so the
  // unpin below runs through the same reconcile a real window has.
  watchSessionPins()
})

beforeEach(async () => {
  setApiRequestConnection(null)
  window.localStorage.clear()
  setConnection({ connectionId: 'local', mode: 'local', baseUrl: 'http://localhost' } as never)
  $sessions.set([row('unrelated')])
  $pinnedSessionIds.set([])
  $profiles.set([profile('default'), profile('work')])
  $activeGatewayProfile.set('default')
  getProfilesMock.mockReset()
  getProfilesMock.mockResolvedValue({ profiles: [profile('default'), profile('work')] })
  await refreshProfiles()
  getSessionMock.mockReset()
  patch.mockClear()
  logLine.mockClear()
  _resetSessionOwnerHintsForTests({ storage: true })
  resetSessionPinMirror()
  resetDeadSessionPrune()
})

afterEach(() => {
  $sessions.set([])
  $pinnedSessionIds.set([])
})

describe('dead-session prune and ids adopted from another connection', () => {
  it("unpins a remote gateway's id from the local key after 404 on every local profile, without a PATCH", async () => {
    // Adopted into the local key by the all-profiles sidebar; its owner is remote.
    $pinnedSessionIds.set(['adopted-remote'])
    await flush()
    setSessionOwnerHint('adopted-remote', { connectionId: 'remote-a', profile: 'default' })
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()
    await flush()

    // Probed on the selected connection, both axes pinned: the owner hint must not reroute it.
    expect(getSessionMock).toHaveBeenCalledWith('adopted-remote', { connectionId: 'local', profile: 'default' })
    expect(getSessionMock).toHaveBeenCalledWith('adopted-remote', { connectionId: 'local', profile: 'work' })
    expect(getSessionMock).not.toHaveBeenCalledWith(
      'adopted-remote',
      expect.objectContaining({ connectionId: 'remote-a' })
    )
    // Gone from this connection's key — the remote gateway's own pin flag is not touched.
    expect($pinnedSessionIds.get()).toEqual([])
    expect(patch).not.toHaveBeenCalled()
  })

  it("leaves the other connection's pin key untouched", async () => {
    window.localStorage.setItem(REMOTE_PIN_KEY, JSON.stringify(['adopted-remote', 'remote-own']))
    $pinnedSessionIds.set(['adopted-remote', 'local-dead'])
    await flush()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()
    await flush()

    expect($pinnedSessionIds.get()).toEqual([])
    expect(JSON.parse(window.localStorage.getItem(REMOTE_PIN_KEY) ?? '[]')).toEqual(['adopted-remote', 'remote-own'])
    expect(patch).not.toHaveBeenCalled()
  })

  it('writes one desktop.log line per unpinned id, naming the connection and the probe basis, never content', async () => {
    $pinnedSessionIds.set(['adopted-remote'])
    await flush()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(logLine).toHaveBeenCalledTimes(1)
    expect(logLine).toHaveBeenCalledWith(
      '[dead-session-prune] unpinned adopted-remote: 404 on 2 profile(s) of connection "local"'
    )
  })

  it('logs nothing when a pin survives (alive elsewhere, or an inconclusive probe)', async () => {
    $pinnedSessionIds.set(['alive-on-work', 'flaky'])
    await flush()
    getSessionMock.mockImplementation(async (id, route) => {
      if (id === 'flaky') {
        throw new Error('gateway restarting')
      }

      if (typeof route === 'object' && route?.profile === 'work') {
        return row(id, { profile: 'work' })
      }

      throw notFound()
    })

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual(['alive-on-work', 'flaky'])
    expect(logLine).not.toHaveBeenCalled()
  })
})
