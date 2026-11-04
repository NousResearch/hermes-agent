// D1+D2 caller capture (packet B): archive, delete and unarchive must route
// the server mutation to the backend that owns the row — decided from the row
// IN HAND, before archive/delete optimistically evicts it. The row-in-hand
// rule: a tagged row pins the write via exactOwnerFor(id, row); an untagged
// row passes explicit null (base routing, immune to a stale resolver hint);
// no row at all leaves the ladder to decide.
//
// Seam: the real sessions API runs unstubbed (so the issued snapshot read and
// PATCH/DELETE are asserted byte-level) while hermesApi itself is captured —
// the same seam as src/api/sessions-owner-routing.test.ts.
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/api/client', async importOriginal => {
  const actual = await importOriginal<typeof ClientApi>()

  return { ...actual, hermesApi: vi.fn() }
})

import * as client from '@/api/client'
import type * as ClientApi from '@/api/client'
import type { HermesApiRequest } from '@/global'
import * as hermes from '@/hermes'
import { type SessionInfo } from '@/hermes'
import { $sessions, setSessions } from '@/store/session'
import { $archivedSessions } from '@/store/sidebar-archive'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  // deleteSession / setSessionArchived stay REAL: the assertions are about the
  // request they issue, not just the args the hook passes.
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

const hermesApi = vi.mocked(client.hermesApi)

type CapturedRequest = HermesApiRequest & { body?: Record<string, unknown> }

const SNAPSHOT = { session_id: 'sess-x', exists: true, runtime_revision: 1, runtime_generation: 1 }

function requests(): CapturedRequest[] {
  return hermesApi.mock.calls.map(([request]) => request as CapturedRequest)
}

function snapshotCalls(): CapturedRequest[] {
  return requests().filter(request => request.path.includes('mutation-snapshot'))
}

function writeCalls(): CapturedRequest[] {
  return requests().filter(request => request.method)
}

function listedRow(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    id: 'sess',
    message_count: 1,
    profile: 'default',
    source: 'desktop',
    started_at: 1,
    title: 'row',
    ...overrides
  } as SessionInfo
}

function legacyRemote(): void {
  client.setApiRequestConnection(null)
  client.setApiRequestLocalMode(false)
}

type Handle = Pick<ReturnType<typeof useSessionActions>, 'archiveSession' | 'removeSession' | 'unarchiveSession'>

function Harness({ onReady }: { onReady: (handle: Handle) => void }) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway: vi.fn().mockResolvedValue(undefined),
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady({
      archiveSession: actions.archiveSession,
      removeSession: actions.removeSession,
      unarchiveSession: actions.unarchiveSession
    })
  }, [actions, onReady])

  return null
}

async function mountHarness(): Promise<Handle> {
  let handle: Handle | undefined
  render(<Harness onReady={h => (handle = h)} />)
  await waitFor(() => expect(handle).toBeDefined())

  return handle as Handle
}

beforeEach(() => {
  setSessions([])
  $archivedSessions.set([])
  hermesApi.mockReset().mockResolvedValue({ ...SNAPSHOT } as never)
  client.setSessionOwnerResolver(null)
  client.setApiRequestConnection(null)
  client.setApiRequestProfile(null)
  client.setApiRequestLocalMode(false)
})

afterEach(() => {
  cleanup()
  setSessions([])
  $archivedSessions.set([])
  client.setSessionOwnerResolver(null)
})

describe('archiveSession owner capture (T9)', () => {
  it('captures the tagged local row before eviction and pins the PATCH back to it under a legacy remote', async () => {
    legacyRemote()
    // Conflicting hint whose PROFILE is eligible: had the owner been resolved
    // AFTER the optimistic eviction (row gone from every slice), this stale
    // hint would win and the archive would leave this machine.
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'default' }))
    setSessions([listedRow({ id: 'local-owned', connection_id: 'local', profile: 'default' })])

    const handle = await mountHarness()

    // The row must already be gone when the RPC starts: the owner was captured
    // from the row BEFORE dropListedSession ran.
    hermesApi.mockImplementation(async request => {
      if (String(request.path).includes('mutation-snapshot')) {
        expect($sessions.get()).toEqual([])
      }

      return { ...SNAPSHOT } as never
    })

    await act(() => handle.archiveSession('local-owned'))

    expect(snapshotCalls()[0]).toMatchObject({ connectionId: 'local', profile: 'default' })
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: 'local', profile: 'default' })
    expect(writeCalls()[0].body).toMatchObject({ archived: true, pinned: false, profile: 'default' })
  })

  it('an untagged row plus a conflicting hint stays byte-identical to base routing', async () => {
    legacyRemote()
    // The hint's profile matches the row's own stamp, so ONLY the explicit
    // null from the row-in-hand rule can keep this on base routing.
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'sage' }))
    setSessions([listedRow({ id: 'untagged', profile: 'sage' })])

    const handle = await mountHarness()

    await act(() => handle.archiveSession('untagged'))

    // Explicit null: the stale hint must not redirect the archive.
    expect(snapshotCalls()[0]).toMatchObject({ connectionId: null, profile: 'sage' })
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: null, profile: 'sage' })
    expect(writeCalls()[0].body).toMatchObject({ archived: true, pinned: false, profile: 'sage' })
  })
})

describe('archiveSession with no row in hand (W1‴ eligibility)', () => {
  it('a resolved profile that conflicts with the ladder owner keeps base routing', async () => {
    legacyRemote()
    // No listed row: the owner comes from the ladder, so the eligibility check
    // must use the RESOLVED profile ('work'), not an empty row-stamp override.
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'tommy' }))
    vi.mocked(hermes.getSession).mockResolvedValue({ ...listedRow({ id: 'rowless', profile: 'work' }) } as never)

    const handle = await mountHarness()

    await act(() => handle.archiveSession('rowless'))

    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: null, profile: 'work' })
    expect(writeCalls()[0].body).toMatchObject({ archived: true, profile: 'work' })
  })
})

describe('removeSession owner capture (K2)', () => {
  it('an untagged row plus a conflicting hint keeps the base DELETE', async () => {
    legacyRemote()
    // Eligible hint profile: only the row-in-hand null keeps base routing.
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'sage' }))
    setSessions([listedRow({ id: 'untagged', profile: 'sage' })])

    const handle = await mountHarness()

    await act(() => handle.removeSession('untagged'))

    expect(snapshotCalls()[0]).toMatchObject({ connectionId: null, profile: 'sage' })
    const del = writeCalls()[0]
    expect(del).toMatchObject({ method: 'DELETE', connectionId: null })
    expect(String(del.path)).toMatch(/^\/api\/sessions\/untagged/)
    expect(String(del.path)).toContain('profile=sage')
  })
})

describe('unarchiveSession owner capture (T10)', () => {
  it('pins the restore PATCH to the tagged archived row backend', async () => {
    client.setApiRequestConnection('reg-R')
    client.setApiRequestLocalMode(false)
    // Conflicting hint with an eligible profile: the row in hand must win.
    client.setSessionOwnerResolver(() => ({ connectionId: 'conn-third', profile: 'op' }))
    $archivedSessions.set([listedRow({ archived: true, id: 'arch-tag', connection_id: 'conn-other', profile: 'op' })])

    const handle = await mountHarness()

    await act(() => handle.unarchiveSession('arch-tag'))

    expect(snapshotCalls()[0]).toMatchObject({ connectionId: 'conn-other', profile: 'op' })
    expect(writeCalls()[0]).toMatchObject({ method: 'PATCH', connectionId: 'conn-other', profile: 'op' })
    expect(writeCalls()[0].body).toMatchObject({ archived: false, profile: 'op' })
  })
})
