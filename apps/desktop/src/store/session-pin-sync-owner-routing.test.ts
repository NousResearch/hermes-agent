// T20 (K4″): pin-sync's writePin must pass the loaded row's owner to the
// PATCH — a tagged row pins the write to its backend; an untagged in-scope row
// passes explicit null (base routing, immune to a stale resolver hint); a
// rowless id (foreign in-set id: loadedRowFor filters other gateways' rows
// out) passes undefined, so the api layer's default ladder lookup resolves the
// owner and routes the write (D2). The removal suppression (no competing unpin
// while a delete/archive RPC is in flight) must survive the change.
//
// Seam: the real pin-sync store and the real setSessionPinnedRemote run; only
// hermesApi is captured (same seam as src/api/sessions-owner-routing.test.ts).
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/api/client', async importOriginal => {
  const actual = await importOriginal<typeof ClientApi>()

  return { ...actual, hermesApi: vi.fn() }
})

import * as client from '@/api/client'
import type * as ClientApi from '@/api/client'
import type { HermesApiRequest } from '@/global'
import { $pinnedSessionIds } from '@/store/layout'
import { $cronSessions, $messagingSessions, $sessions, setSessions } from '@/store/session'
import { beginSessionMutation } from '@/store/session-removal'
import type { SessionInfo } from '@/types/hermes'

import { resetSessionPinMirror, watchSessionPins } from './session-pin-sync'

const hermesApi = vi.mocked(client.hermesApi)

const SNAPSHOT = { session_id: 'sess-x', exists: true, runtime_revision: 1, runtime_generation: 1 }

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

type CapturedRequest = HermesApiRequest & { body?: Record<string, unknown> }

function writeCalls(): CapturedRequest[] {
  return hermesApi.mock.calls.map(([request]) => request as CapturedRequest).filter(request => request.method)
}

const settle = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0))

beforeAll(() => {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}
  // Attach the listeners once — module state is process-global.
  watchSessionPins()
})

beforeEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $pinnedSessionIds.set([])
  resetSessionPinMirror()
  hermesApi.mockReset().mockResolvedValue({ ...SNAPSHOT } as never)
  client.setSessionOwnerResolver(null)
  client.setApiRequestConnection(null)
  client.setApiRequestProfile(null)
  client.setApiRequestLocalMode(false)
})

afterEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $pinnedSessionIds.set([])
  client.setSessionOwnerResolver(null)
})

function registryAmbient(): void {
  client.setApiRequestConnection('reg-R')
  client.setApiRequestLocalMode(false)
}

describe('writePin owner routing (T20)', () => {
  it('a tagged row whose owner IS the ambient connection stays byte-identical to base', async () => {
    registryAmbient()
    setSessions([row('a1', { connection_id: 'reg-R', profile: 'rp' })])

    $pinnedSessionIds.set(['a1'])
    await settle()

    expect(writeCalls()).toHaveLength(1)
    expect(writeCalls()[0]).toMatchObject({
      method: 'PATCH',
      connectionId: 'reg-R',
      profile: 'rp',
      path: '/api/sessions/a1'
    })
    expect(writeCalls()[0].body).toMatchObject({ pinned: true, profile: 'rp' })
  })

  it('an untagged in-scope row plus a conflicting hint keeps the base PATCH', async () => {
    registryAmbient()
    // The hint's profile matches the row's own stamp, so ONLY the explicit
    // null from the row-in-hand rule can keep this PATCH on base routing.
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'rp' }))
    setSessions([row('u1', { profile: 'rp' })])

    $pinnedSessionIds.set(['u1'])
    await settle()

    expect(writeCalls()).toHaveLength(1)
    // Explicit null from the row in hand: the stale hint must not redirect.
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: 'reg-R', profile: 'rp' })
    expect(writeCalls()[0].body).toMatchObject({ pinned: true, profile: 'rp' })
  })

  it('a rowless unpin for an in-set id routes by the ladder owner (K4″ foreign id)', async () => {
    registryAmbient()
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'op' }))

    // Pin while the row is loaded and in scope (base PATCH: owner == ambient).
    setSessions([row('f1', { connection_id: 'reg-R', profile: 'rp' })])
    $pinnedSessionIds.set(['f1'])
    await settle()
    expect(writeCalls()).toHaveLength(1)
    hermesApi.mockClear()

    // The id leaves the pin scope (the only loaded row is now another
    // gateway's twin), so loadedRowFor finds nothing: writePin must pass
    // owner undefined and let the ladder route the unpin to the owner.
    setSessions([row('f1', { connection_id: 'conn-foreign', profile: 'x' })])
    $pinnedSessionIds.set([])
    await settle()

    expect(writeCalls()).toHaveLength(1)
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: 'conn-other', profile: 'op' })
    expect(writeCalls()[0].body).toMatchObject({ pinned: false, profile: 'op' })
  })

  it('a pending removal still suppresses the competing unpin', async () => {
    registryAmbient()
    setSessions([row('r1', { connection_id: 'reg-R', profile: 'rp' })])
    $pinnedSessionIds.set(['r1'])
    await settle()
    expect(writeCalls()).toHaveLength(1)
    hermesApi.mockClear()

    // The delete/archive RPC owns the durable write; no unpin may race it.
    beginSessionMutation(['r1'])
    $pinnedSessionIds.set([])
    await settle()

    expect(writeCalls()).toHaveLength(0)
  })
})
