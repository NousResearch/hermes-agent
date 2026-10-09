import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type * as HermesModule from '@/hermes'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { $activeGatewayProfile } from '@/store/profile'
import { markSessionGone, resetBackgroundPollingGuard, resetRuntimeGoneHealing } from '@/store/runtime-gone'
import { $connection, setSessions } from '@/store/session'
import {
  $sessionTiles,
  closeSessionTile,
  openSessionTile,
  patchSessionTile,
  sessionTileDelegate
} from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getLatestSessionMessages: vi.fn(),
  getSession: vi.fn()
}))
vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')
const { requestGatewayForProfile } = await import('@/store/gateway')

// Shapes from the incident: a tile on the Mac-local source whose turn finished
// on the backend while the window was switched to a remote source.
const STORED_ID = '20261008_195234_dd1c54'
const STALE_RUNTIME = '173890a7'
const FINAL = 'FINAL ANSWER persisted while the window was on another source'

type Connection = NonNullable<ReturnType<typeof $connection.get>>

const LOCAL = { baseUrl: '', connectionId: 'local', isFullscreen: false, mode: 'local' } as Connection

const REMOTE = {
  baseUrl: 'http://100.98.87.106:9119',
  connectionId: 'agents-server',
  isFullscreen: false,
  mode: 'remote'
} as Connection

const row = {
  ended_at: null,
  id: STORED_ID,
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 4,
  model: null,
  output_tokens: 0,
  preview: null,
  profile: 'default',
  source: null,
  started_at: 0,
  title: 'Carte Blanche website revamp plan'
} as SessionInfo

function staleStreamingSnapshot(): ClientSessionState {
  return {
    ...createClientSessionState(STORED_ID, [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'first question' }] },
      { id: 'a1', role: 'assistant', parts: [{ type: 'text', text: 'first answer' }] },
      { id: 'u2', role: 'user', parts: [{ type: 'text', text: 'second question' }] },
      {
        id: `assistant-stream-${STALE_RUNTIME}`,
        role: 'assistant',
        pending: true,
        parts: [{ type: 'text', text: 'partial tool commentary' }]
      }
    ]),
    busy: true,
    turnLive: true,
    streamId: `assistant-stream-${STALE_RUNTIME}`
  }
}

function mountDelegate() {
  const runtimeIdByStoredSessionIdRef = { current: new Map([[STORED_ID, STALE_RUNTIME]]) }
  const sessionStateByRuntimeIdRef = { current: new Map([[STALE_RUNTIME, staleStreamingSnapshot()]]) }

  const updateSessionState = vi.fn(
    (runtimeId: string, updater: (state: ClientSessionState) => ClientSessionState, storedSessionId?: string) => {
      const previous = sessionStateByRuntimeIdRef.current.get(runtimeId) ?? createClientSessionState(storedSessionId)
      const next = updater(previous)
      sessionStateByRuntimeIdRef.current.set(runtimeId, next)

      return next
    }
  )

  renderHook(() =>
    useSessionTileDelegate({
      archiveSession: vi.fn(async () => undefined),
      branchLoadedSession: vi.fn(async () => false) as never,
      branchStoredSession: vi.fn(async () => undefined),
      executeSlashCommand: vi.fn(async () => undefined) as never,
      removeSession: vi.fn(async () => undefined),
      requestGateway: vi.fn(async () => ({})) as never,
      runtimeIdByStoredSessionIdRef: runtimeIdByStoredSessionIdRef as never,
      sessionStateByRuntimeIdRef: sessionStateByRuntimeIdRef as never,
      updateSessionState: updateSessionState as never
    })
  )

  return { sessionStateByRuntimeIdRef }
}

const transcriptText = (state: ClientSessionState | undefined) => (state?.messages ?? []).map(chatMessageText)

describe('session tile after a source round-trip', () => {
  beforeEach(() => {
    localStorage.clear()
    $activeGatewayProfile.set('default')
    $connection.set(LOCAL)
    setSessions([row])
    vi.mocked(getLatestSessionMessages).mockReset()
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      session_id: STORED_ID,
      messages: [
        { id: 1, role: 'user', content: 'first question', timestamp: 1 },
        { id: 2, role: 'assistant', content: 'first answer', timestamp: 2 },
        { id: 3, role: 'user', content: 'second question', timestamp: 3 },
        { id: 4, role: 'assistant', content: FINAL, timestamp: 4 }
      ]
    } as never)
    vi.mocked(requestGatewayForProfile).mockReset()
    // The runtime the tile streamed on was detached when the source's socket
    // was pruned and reaped once its turn finished; a resume mints a new one.
    vi.mocked(requestGatewayForProfile).mockImplementation(async (_profile, method) =>
      method === 'session.resume'
        ? ({ session_id: 'fresh-runtime', resumed: STORED_ID, info: { running: false } } as never)
        : ({} as never)
    )
  })

  afterEach(() => {
    cleanup()
    $connection.set(LOCAL)
    $activeGatewayProfile.set('default')
    closeSessionTile(STORED_ID)
    setSessions([])
    resetRuntimeGoneHealing()
    resetBackgroundPollingGuard()
    localStorage.clear()
  })

  function openStreamingTile() {
    openSessionTile(STORED_ID)
    patchSessionTile(STORED_ID, { runtimeId: STALE_RUNTIME })
    expect($sessionTiles.get().find(tile => tile.storedSessionId === STORED_ID)?.runtimeId).toBe(STALE_RUNTIME)
  }

  async function resumeSwappedInTile() {
    // The swapped-in tile is runtime-less, so it mounts and asks to resume.
    expect($sessionTiles.get().find(tile => tile.storedSessionId === STORED_ID)?.runtimeId).toBeUndefined()

    return sessionTileDelegate()!.resumeTile(STORED_ID)
  }

  it('re-resumes from the backend instead of repainting the pre-switch streaming snapshot (connection switch)', async () => {
    openStreamingTile()
    const { sessionStateByRuntimeIdRef } = mountDelegate()

    // Switch to the remote source, then back. Each commit wipes the lists;
    // the turn finishes on the local backend in between, with no socket to
    // deliver its events.
    wipeSessionListsForGatewaySwitch()
    $connection.set(REMOTE)
    wipeSessionListsForGatewaySwitch()
    $connection.set(LOCAL)
    setSessions([row])

    const runtimeId = await resumeSwappedInTile()

    expect(runtimeId).toBe('fresh-runtime')
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'default',
      'session.resume',
      expect.objectContaining({ session_id: STORED_ID }),
      undefined,
      undefined
    )

    const shown = sessionStateByRuntimeIdRef.current.get(runtimeId)
    expect(transcriptText(shown)).toContain(FINAL)
    expect(shown?.busy).toBe(false)
  })

  it('re-resumes after a profile round-trip too', async () => {
    openStreamingTile()
    const { sessionStateByRuntimeIdRef } = mountDelegate()

    $activeGatewayProfile.set('other-profile')
    $activeGatewayProfile.set('default')

    const runtimeId = await resumeSwappedInTile()

    expect(runtimeId).toBe('fresh-runtime')
    expect(transcriptText(sessionStateByRuntimeIdRef.current.get(runtimeId))).toContain(FINAL)
  })

  it('shows the persisted reply when the resume reattaches the same parked runtime, now idle', async () => {
    openStreamingTile()
    const { sessionStateByRuntimeIdRef } = mountDelegate()

    // The backend kept the detached runtime parked instead of reaping it, so
    // the resume hands back the very id this cache holds a frozen
    // half-streamed snapshot for.
    vi.mocked(requestGatewayForProfile).mockImplementation(async (_profile, method) =>
      method === 'session.resume'
        ? ({ session_id: STALE_RUNTIME, resumed: STORED_ID, info: { running: false } } as never)
        : ({} as never)
    )

    wipeSessionListsForGatewaySwitch()
    $connection.set(REMOTE)
    wipeSessionListsForGatewaySwitch()
    $connection.set(LOCAL)
    setSessions([row])

    const runtimeId = await resumeSwappedInTile()

    expect(runtimeId).toBe(STALE_RUNTIME)

    const shown = sessionStateByRuntimeIdRef.current.get(runtimeId)
    expect(transcriptText(shown)).toContain(FINAL)
    expect(transcriptText(shown)).not.toContain('partial tool commentary')
    expect(shown?.busy).toBe(false)
  })

  it('never re-binds a runtime the gateway declared gone', async () => {
    openStreamingTile()
    const { sessionStateByRuntimeIdRef } = mountDelegate()

    // A status poll for the tile's runtime came back 4001 "not in memory".
    markSessionGone(STALE_RUNTIME)
    expect($sessionTiles.get().find(tile => tile.storedSessionId === STORED_ID)?.runtimeId).toBeUndefined()

    // The unbound tile re-resumes; the cache's reverse entry must not hand the
    // dead id straight back.
    const runtimeId = await sessionTileDelegate()!.resumeTile(STORED_ID)

    expect(runtimeId).toBe('fresh-runtime')
    expect(transcriptText(sessionStateByRuntimeIdRef.current.get(runtimeId))).toContain(FINAL)
  })
})
