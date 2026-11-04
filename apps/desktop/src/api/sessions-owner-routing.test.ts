import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Real client module except hermesApi, so exactOwnerFor / ownerConnectionPin /
// the three-state local mode and the owner resolver run unstubbed while every
// issued request is captured.
vi.mock('./client', async importOriginal => {
  const actual = await importOriginal<typeof ClientApi>()

  return { ...actual, hermesApi: vi.fn() }
})

import type { HermesApiRequest } from '@/global'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import type * as ClientApi from './client'
import * as client from './client'
import {
  deleteSession,
  renameSession,
  setSessionArchived,
  setSessionPinnedRemote,
  setSessionUnreadRemote
} from './sessions'

const hermesApi = vi.mocked(client.hermesApi)

type CapturedRequest = HermesApiRequest & { body?: Record<string, unknown> }

const SNAPSHOT = { session_id: 'sess-x', exists: true, runtime_revision: 1, runtime_generation: 1 }
const OWNER_LOCAL = { connectionId: 'local', profile: 'default' }

function requests(): CapturedRequest[] {
  return hermesApi.mock.calls.map(([request]) => request as CapturedRequest)
}

function snapshotCalls(): CapturedRequest[] {
  return requests().filter(request => request.path.includes('mutation-snapshot'))
}

function writeCalls(): CapturedRequest[] {
  return requests().filter(request => request.method)
}

function bodyOf(request: CapturedRequest): Record<string, unknown> {
  return (request.body ?? {}) as Record<string, unknown>
}

function legacyRemote() {
  client.setApiRequestConnection(null)
  client.setApiRequestLocalMode(false)
}

beforeEach(() => {
  vi.clearAllMocks()
  hermesApi.mockResolvedValue({ ...SNAPSHOT } as never)
  client.setSessionOwnerResolver(null)
  client.setApiRequestConnection(null)
  client.setApiRequestProfile(null)
})

afterEach(() => {
  $sessions.set([])
})

describe('T2 boot window write cells (mode unknown; runs first in file order)', () => {
  it('a non-local exact owner is pinned during boot (new behaviour, R-d); a local owner stays base', async () => {
    // No setApiRequestLocalMode call anywhere before this test: the module
    // mode is still null (boot window). Vitest runs a file's tests in
    // declaration order (sequence.shuffle is off in this repo).
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'op' }))
    await setSessionUnreadRemote('sess-boot', true)
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: 'conn-other', profile: 'op' })

    hermesApi.mockClear()
    client.setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'lp' }))
    await setSessionUnreadRemote('sess-boot2', true)
    expect(writeCalls()[0].method).toBe('PATCH')
    expect(writeCalls()[0].connectionId).toBeNull()
    expect(writeCalls()[0].profile).toBeUndefined()
    expect(writeCalls()[0].path).toBe('/api/sessions/sess-boot2')
  })
})

describe('T2 write topology table', () => {
  const owners: Record<string, (() => unknown) | null> = {
    none: null,
    bare: () => 'bare-prof',
    local: () => ({ connectionId: 'local', profile: 'lp' }),
    R: () => ({ connectionId: 'reg-R', profile: 'rp' }),
    other: () => ({ connectionId: 'conn-other', profile: 'op' })
  }

  // Expected connectionId/profile on the issued write per primary × owner.
  // 'none' and 'bare' never pin (byte-identical to base); an owner equal to
  // the ambient connection never pins; legacy remote pins every exact owner.
  const cases: Array<[string, string, string | null, string | undefined]> = [
    ['local', 'none', null, undefined],
    ['local', 'bare', null, undefined],
    ['local', 'local', null, undefined],
    ['local', 'R', 'reg-R', 'rp'],
    ['local', 'other', 'conn-other', 'op'],
    ['registry', 'none', 'reg-R', undefined],
    ['registry', 'bare', 'reg-R', undefined],
    ['registry', 'local', 'local', 'lp'],
    ['registry', 'R', 'reg-R', undefined],
    ['registry', 'other', 'conn-other', 'op'],
    ['legacy', 'none', null, undefined],
    ['legacy', 'bare', null, undefined],
    ['legacy', 'local', 'local', 'lp'],
    ['legacy', 'R', 'reg-R', 'rp'],
    ['legacy', 'other', 'conn-other', 'op']
  ]

  for (const [primary, ownerKey, expectedConnection, expectedProfile] of cases) {
    it(`${primary} primary × ${ownerKey} owner`, async () => {
      if (primary === 'local') {
        client.setApiRequestConnection(null)
        client.setApiRequestLocalMode(true)
      } else if (primary === 'registry') {
        client.setApiRequestConnection('reg-R')
        client.setApiRequestLocalMode(false)
      } else {
        legacyRemote()
      }

      client.setSessionOwnerResolver(owners[ownerKey] as never)
      await setSessionUnreadRemote('sess-t2w', true)
      const write = writeCalls()[0]

      expect(write.method).toBe('PATCH')
      expect(write.connectionId).toBe(expectedConnection)
      expect(write.profile).toBe(expectedProfile)
      expect(bodyOf(write)).toMatchObject({ unread: true })

      if (expectedProfile) {
        expect(bodyOf(write)).toMatchObject({ profile: expectedProfile })
      } else {
        expect(bodyOf(write)).not.toHaveProperty('profile')
      }

      // A PATCH carries the profile in its body, never in the URL.
      expect(write.path).toBe('/api/sessions/sess-t2w')
      // The single scope also feeds the snapshot read.
      expect(snapshotCalls()[0].connectionId).toBe(expectedConnection)
      expect(snapshotCalls()[0].profile).toBe(expectedProfile)
    })
  }

  it('mid-rehome (registry primary, tag cleared, mode still remote): an exact local owner pins to local (R-e)', async () => {
    legacyRemote()
    await setSessionUnreadRemote('sess-re', true, undefined, OWNER_LOCAL, '')
    expect(writeCalls()[0]).toMatchObject({ connectionId: 'local', profile: 'default' })
  })
})

describe('T1 legacy remote, local-owned row (K1-style caller)', () => {
  it('archive/unarchive/pin/unread/rename all route to {connectionId:local, profile:default}; ambient never leaks', async () => {
    legacyRemote()
    client.setApiRequestProfile('work')

    const edits: Array<[string, () => Promise<unknown>]> = [
      ['archive', () => setSessionArchived('sess-t1', true, 'default', OWNER_LOCAL, '')],
      ['unarchive', () => setSessionArchived('sess-t1', false, 'default', OWNER_LOCAL, '')],
      ['pin', () => setSessionPinnedRemote('sess-t1', true, 'default', OWNER_LOCAL, '')],
      ['unread', () => setSessionUnreadRemote('sess-t1', true, 'default', OWNER_LOCAL, '')],
      ['rename', () => renameSession('sess-t1', 'Renamed', 'default', OWNER_LOCAL, '')]
    ]

    for (const [name, run] of edits) {
      hermesApi.mockClear()
      await run()
      expect(snapshotCalls(), name).toHaveLength(1)
      expect(snapshotCalls()[0], name).toEqual({
        connectionId: 'local',
        profile: 'default',
        path: '/api/sessions/sess-t1/mutation-snapshot?profile=default'
      })
      expect(writeCalls(), name).toHaveLength(1)
      expect(writeCalls()[0], name).toMatchObject({
        method: 'PATCH',
        connectionId: 'local',
        profile: 'default',
        path: '/api/sessions/sess-t1',
        body: { profile: 'default' }
      })

      for (const request of requests()) {
        expect(JSON.stringify(request), name).not.toContain('work')
      }
    }
  })

  it('delete via K2 keeps the caller object scope verbatim even against a conflicting resolver', async () => {
    legacyRemote()
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'op' }))
    await deleteSession('sess-t1', { connectionId: 'local', profile: 'default' })
    expect(snapshotCalls()[0]).toEqual({
      connectionId: 'local',
      profile: 'default',
      path: '/api/sessions/sess-t1/mutation-snapshot?profile=default'
    })
    expect(writeCalls()[0]).toMatchObject({ method: 'DELETE', connectionId: 'local', profile: 'default' })
    expect(writeCalls()[0].path).toContain('/api/sessions/sess-t1?')
    expect(writeCalls()[0].path).toContain('profile=default')
  })
})

describe('T3 DELETE stays byte-identical to base when unpinned', () => {
  it('no owner, no profile: unscoped snapshot + DELETE with identity params only', async () => {
    await deleteSession('sess-d')
    expect(snapshotCalls()[0]).toEqual({ connectionId: null, path: '/api/sessions/sess-d/mutation-snapshot' })
    const write = writeCalls()[0]

    expect(write.method).toBe('DELETE')
    expect(write.connectionId).toBeNull()
    expect(write.profile).toBeUndefined()
    expect(write.path).toMatch(/^\/api\/sessions\/sess-d\?request_id=/)
    expect(write.body).toBeUndefined()
  })

  it('explicit null owner with a resolver answer: base construction', async () => {
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'op' }))
    await deleteSession('sess-d', 'work', null)
    expect(snapshotCalls()[0]).toEqual({
      connectionId: null,
      profile: 'work',
      path: '/api/sessions/sess-d/mutation-snapshot?profile=work'
    })
    expect(writeCalls()[0]).toMatchObject({ method: 'DELETE', connectionId: null, profile: 'work' })
  })

  it('bare-string resolver owner: no pin, base construction', async () => {
    client.setSessionOwnerResolver(() => 'bare-prof')
    await deleteSession('sess-d')
    expect(writeCalls()[0]).toMatchObject({ method: 'DELETE', connectionId: null })
    expect(writeCalls()[0].path).toMatch(/^\/api\/sessions\/sess-d\?request_id=/)
  })

  it('profile-only object scope: sessionScoped base output, never ambient-filled', async () => {
    client.setApiRequestProfile('work')
    await deleteSession('sess-d', { profile: 'work' })
    expect(snapshotCalls()[0]).toEqual({
      connectionId: null,
      profile: 'work',
      path: '/api/sessions/sess-d/mutation-snapshot?profile=work'
    })
  })

  it('ambient profile never leaks into an unfilled DELETE', async () => {
    client.setApiRequestProfile('work')
    await deleteSession('sess-d')
    expect(snapshotCalls()[0].path).toBe('/api/sessions/sess-d/mutation-snapshot')
    expect(writeCalls()[0].path).not.toContain('profile=work')
  })
})

describe('T4 caller-authoritative object scope', () => {
  it('an object scope with a connectionId wins over the owner pin (K2)', async () => {
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'op' }))
    await deleteSession('sess-t4', { connectionId: 'hermes-pi', profile: 'tommy' })
    expect(snapshotCalls()[0]).toEqual({
      connectionId: 'hermes-pi',
      profile: 'tommy',
      path: '/api/sessions/sess-t4/mutation-snapshot?profile=tommy'
    })
    expect(writeCalls()[0]).toMatchObject({ method: 'DELETE', connectionId: 'hermes-pi', profile: 'tommy' })
  })
})

describe('T5 explicit conflicting profile refuses the pin', () => {
  it('positional profile that names another profile ⇒ exact base construction', async () => {
    legacyRemote()
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-b', profile: 'tommy' }))
    await setSessionArchived('sess-c', true, 'work')
    expect(snapshotCalls()[0]).toEqual({
      connectionId: null,
      profile: 'work',
      path: '/api/sessions/sess-c/mutation-snapshot?profile=work'
    })
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: null, profile: 'work' })
    expect(bodyOf(writeCalls()[0])).toMatchObject({ archived: true, pinned: false, profile: 'work' })
  })
})

describe('T6 owner hint targetProfile', () => {
  it('the pinned scope names the owner targetProfile when the caller profile matches it', async () => {
    legacyRemote()
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-b', profile: 'desk', targetProfile: 'work' }))
    await setSessionPinnedRemote('sess-h', true, 'work')
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: 'conn-b', profile: 'work' })
    expect(writeCalls()[0].path).toBe('/api/sessions/sess-h')
    expect(bodyOf(writeCalls()[0])).toMatchObject({ pinned: true, profile: 'work' })
  })

  it('an empty caller profile (row rung, K1-style) is eligible', async () => {
    legacyRemote()
    await setSessionUnreadRemote('sess-h', true, undefined, { connectionId: 'conn-b', profile: 'tommy' }, '')
    expect(writeCalls()[0]).toMatchObject({ connectionId: 'conn-b', profile: 'tommy' })
  })
})

describe('T7 unstubbed owner ladder (real knownOwnerForSession)', () => {
  it('a stamped $sessions row routes the write through the real resolver registration', async () => {
    // Importing session-states registers the REAL knownOwnerForSession ladder
    // via setSessionOwnerResolver (store/session-states.ts). Its module init
    // also touches the ambient request scope, so re-pin it afterwards.
    await import('@/store/session-states')
    legacyRemote()
    client.setApiRequestProfile(null)
    $sessions.set([{ id: 'sess-rig', connection_id: ' conn-rig ', profile: 'rigprof' } as SessionInfo])
    await setSessionUnreadRemote('sess-rig', true)
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: 'conn-rig', profile: 'rigprof' })

    hermesApi.mockClear()
    $sessions.set([{ id: 'sess-bare', profile: 'bareprof' } as SessionInfo])
    await setSessionUnreadRemote('sess-bare', true)
    expect(writeCalls()[0].connectionId).toBeNull()
    expect(writeCalls()[0].profile).toBeUndefined()
  })
})

describe('T12 pending snapshot, then the ambient scope moves', () => {
  it('the write keeps the captured destination', async () => {
    legacyRemote()
    let releaseSnapshot!: () => void
    hermesApi.mockImplementation(async request => {
      if (request.path.includes('mutation-snapshot')) {
        await new Promise<void>(resolve => {
          releaseSnapshot = resolve
        })

        return { session_id: 'sess-t12', exists: true, runtime_revision: 4, runtime_generation: 2 } as never
      }

      return { ok: true } as never
    })
    const pending = setSessionUnreadRemote('sess-t12', true, 'default', OWNER_LOCAL, '')
    await Promise.resolve()
    client.setApiRequestConnection('moved-tag')
    client.setApiRequestProfile('movedprof')
    releaseSnapshot()
    await pending
    expect(writeCalls()[0]).toMatchObject({ connectionId: 'local', profile: 'default' })

    for (const request of requests()) {
      expect(JSON.stringify(request)).not.toContain('moved')
    }
  })
})

describe('T13 lost-ACK retained identity', () => {
  it('the same destination reuses the retained identity; a different destination re-reads', async () => {
    legacyRemote()
    hermesApi.mockImplementation(async request => {
      if (request.path.includes('mutation-snapshot')) {
        return { session_id: 'sess-t13', exists: true, runtime_revision: 7, runtime_generation: 2 } as never
      }

      throw new Error('network: connection lost')
    })
    const OWNER_A = { connectionId: 'conn-a', profile: 'pa' }
    const OWNER_B = { connectionId: 'conn-b', profile: 'pb' }

    await expect(setSessionPinnedRemote('sess-t13', true, undefined, OWNER_A)).rejects.toThrow('connection lost')
    await expect(setSessionPinnedRemote('sess-t13', true, undefined, OWNER_A)).rejects.toThrow('connection lost')
    expect(snapshotCalls()).toHaveLength(1)
    expect(bodyOf(writeCalls()[1]).request_id).toBe(bodyOf(writeCalls()[0]).request_id)
    expect(writeCalls()[1].connectionId).toBe('conn-a')

    await expect(setSessionPinnedRemote('sess-t13', true, undefined, OWNER_B)).rejects.toThrow('connection lost')
    expect(snapshotCalls()).toHaveLength(2)
    expect(bodyOf(writeCalls()[2]).request_id).not.toBe(bodyOf(writeCalls()[0]).request_id)
    expect(writeCalls()[2].connectionId).toBe('conn-b')
  })
})

describe('T14 MissingMutationSnapshotRoute legacy fallback', () => {
  it('keeps the pinned scope on the fallback write with no identity params', async () => {
    legacyRemote()
    hermesApi.mockImplementation(async request => {
      if (request.path.includes('mutation-snapshot')) {
        throw new Error('404: {"detail":"No such API endpoint: /api/sessions/sess-t14/mutation-snapshot"}')
      }

      return { ok: true } as never
    })
    await setSessionUnreadRemote('sess-t14', true, undefined, OWNER_LOCAL, '')
    // The snapshot GET was attempted (and failed unrouted), then the legacy
    // fallback write went out with the SAME pinned scope and no identity.
    expect(snapshotCalls()).toHaveLength(1)
    expect(writeCalls()).toHaveLength(1)
    expect(writeCalls()[0]).toMatchObject({
      method: 'PATCH',
      connectionId: 'local',
      profile: 'default',
      path: '/api/sessions/sess-t14'
    })
    expect(writeCalls()[0].body).toEqual({ unread: true, profile: 'default' })
  })
})

describe('T21 profile fallback (owner == ambient keeps base; legacy remote pins the row profile)', () => {
  it('row with connection_id but no profile: resolved default stays default under local and registry primaries', async () => {
    client.setApiRequestProfile('work')
    const ownerFromRow = client.exactOwnerFor('sess-t21', { connection_id: 'local' })
    expect(ownerFromRow).toEqual({ connectionId: 'local', profile: 'default' })

    client.setApiRequestConnection(null)
    client.setApiRequestLocalMode(true)
    await setSessionUnreadRemote('sess-t21', true, 'default', ownerFromRow, '')
    expect(writeCalls()[0]).toEqual({
      connectionId: null,
      profile: 'default',
      method: 'PATCH',
      path: '/api/sessions/sess-t21',
      body: {
        unread: true,
        profile: 'default',
        request_id: expect.any(String),
        expected_revision: 1,
        expected_generation: 1
      }
    })

    const ownerFromRegistryRow = client.exactOwnerFor('sess-t21', { connection_id: 'reg-R' })
    expect(ownerFromRegistryRow).toEqual({ connectionId: 'reg-R', profile: 'default' })
    hermesApi.mockClear()
    client.setApiRequestConnection('reg-R')
    client.setApiRequestLocalMode(false)
    await setSessionUnreadRemote('sess-t21', true, 'default', ownerFromRegistryRow, '')
    expect(writeCalls()[0]).toEqual({
      connectionId: 'reg-R',
      profile: 'default',
      method: 'PATCH',
      path: '/api/sessions/sess-t21',
      body: {
        unread: true,
        profile: 'default',
        request_id: expect.any(String),
        expected_revision: 1,
        expected_generation: 1
      }
    })
  })

  it('legacy remote with a local owner pins {connectionId:local, profile:default}; ambient work never leaks', async () => {
    client.setApiRequestProfile('work')
    const ownerFromRow = client.exactOwnerFor('sess-t21', { connection_id: 'local' })
    legacyRemote()
    await setSessionUnreadRemote('sess-t21', true, 'default', ownerFromRow, '')
    expect(writeCalls()[0]).toMatchObject({ connectionId: 'local', profile: 'default' })

    for (const request of requests()) {
      expect(JSON.stringify(request)).not.toContain('work')
    }
  })
})
