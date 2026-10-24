// Regression: deleting a session from the Archived filter view left a ghost
// row. Archived rows live in $archivedSessions (their own capped store —
// they're excluded from $sessions by design), and removeSession only pruned
// $sessions. The ghost row then resumed into a hard-deleted id: resume 404 →
// goneSessionVerdict saw the row still listed → 'retry' → an unrecoverable
// spinner (Aug 2026 desktop audit, F8).
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deleteSession, type SessionInfo } from '@/hermes'
import { $sessions, setSessions } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { $archivedSessions } from '@/store/sidebar-archive'
import { dropTranscriptTailEverywhere } from '@/store/transcript-tail-cache'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/transcript-tail-cache', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  dropTranscriptTailEverywhere: vi.fn()
}))

function archivedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: true,
    ended_at: null,
    id: 'arch-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'archived ghost',
    tool_call_count: 0,
    ...overrides
  } as SessionInfo
}

type Handle = Pick<ReturnType<typeof useSessionActions>, 'removeSession'>

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
    onReady({ removeSession: actions.removeSession })
  }, [actions, onReady])

  return null
}

async function mountHarness(): Promise<Handle> {
  let handle: Handle | undefined
  render(<Harness onReady={h => (handle = h)} />)
  await waitFor(() => expect(handle).toBeDefined())

  return handle as Handle
}

describe('removeSession × archived view store', () => {
  beforeEach(() => {
    setSessions([])
    $archivedSessions.set([])
    vi.mocked(deleteSession).mockReset()
    vi.mocked(dropTranscriptTailEverywhere).mockReset()
    $removedSessionIds.set(new Set())
  })

  afterEach(() => {
    cleanup()
    setSessions([])
    $archivedSessions.set([])
    $removedSessionIds.set(new Set())
    vi.restoreAllMocks()
  })

  it('evicts the row from $archivedSessions on successful delete', async () => {
    $archivedSessions.set([archivedSession(), archivedSession({ id: 'arch-2', title: 'keep me' })])
    vi.mocked(deleteSession).mockResolvedValue({ ok: true })

    const handle = await mountHarness()

    await act(() => handle.removeSession('arch-1'))

    expect(vi.mocked(deleteSession)).toHaveBeenCalledWith('arch-1', undefined)
    expect($archivedSessions.get().map(session => session.id)).toEqual(['arch-2'])
  })

  it('restores the archived row when the delete RPC fails', async () => {
    $archivedSessions.set([archivedSession()])
    vi.mocked(deleteSession).mockRejectedValue(new Error('backend down'))

    const handle = await mountHarness()

    await act(() => handle.removeSession('arch-1'))

    expect($archivedSessions.get().map(session => session.id)).toEqual(['arch-1'])
    expect($removedSessionIds.get().has('arch-1')).toBe(false)
  })

  it('keeps a committed deletion evicted when transcript cleanup throws', async () => {
    const session = archivedSession({ archived: false, profile: 'default' })
    setSessions([session])
    $archivedSessions.set([archivedSession({ id: 'keep-archived' })])
    vi.mocked(deleteSession).mockResolvedValue({ ok: true })
    vi.mocked(dropTranscriptTailEverywhere).mockImplementation(() => {
      throw new Error('cache unavailable')
    })
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    const handle = await mountHarness()
    await act(() => handle.removeSession(session.id))

    expect(deleteSession).toHaveBeenCalledWith(session.id, 'default')
    expect($sessions.get()).toEqual([])
    expect($removedSessionIds.get().has(session.id)).toBe(true)
    expect($archivedSessions.get().map(row => row.id)).toEqual(['keep-archived'])
  })

  it('passes the archived row profile to deleteSession', async () => {
    $archivedSessions.set([archivedSession({ profile: 'sage' } as Partial<SessionInfo>)])
    vi.mocked(deleteSession).mockResolvedValue({ ok: true })

    const handle = await mountHarness()

    await act(() => handle.removeSession('arch-1'))

    expect(vi.mocked(deleteSession)).toHaveBeenCalledWith('arch-1', 'sage')
  })
})
