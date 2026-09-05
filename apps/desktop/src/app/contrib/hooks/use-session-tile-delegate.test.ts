import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { clearSessionDraft, stashSessionDraft } from '@/store/composer'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { setSessionOwnerHint, setSessions } from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: '' }))
}))
vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')
const { requestGatewayForAgent, requestGatewayForProfile } = await import('@/store/gateway')

const row = (over: Partial<SessionInfo>): SessionInfo =>
  ({
    ended_at: null,
    id: 'live',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: null,
    started_at: 0,
    title: null,
    ...over
  }) as SessionInfo

function renderTile(
  requestGateway: ReturnType<typeof vi.fn>,
  refs?: {
    runtimeIdByStoredSessionIdRef?: { current: Map<string, string> }
    sessionStateByRuntimeIdRef?: { current: Map<string, unknown> }
    updateSessionState?: ReturnType<typeof vi.fn>
  }
) {
  renderHook(() =>
    useSessionTileDelegate({
      archiveSession: vi.fn(async () => undefined),
      branchStoredSession: vi.fn(async () => undefined),
      executeSlashCommand: vi.fn(async () => undefined) as never,
      removeSession: vi.fn(async () => undefined),
      requestGateway: requestGateway as never,
      runtimeIdByStoredSessionIdRef: (refs?.runtimeIdByStoredSessionIdRef ?? { current: new Map() }) as never,
      sessionStateByRuntimeIdRef: (refs?.sessionStateByRuntimeIdRef ?? { current: new Map() }) as never,
      updateSessionState: (refs?.updateSessionState ?? vi.fn()) as never
    })
  )
}

describe('useSessionTileDelegate resumeTile', () => {
  beforeEach(() => {
    setSessions([])
    vi.mocked(getLatestSessionMessages).mockClear()
    vi.mocked(requestGatewayForAgent).mockReset()
    vi.mocked(requestGatewayForProfile).mockReset()
  })

  afterEach(() => {
    setSessions([])
  })

  it('carries the owning profile into a cold tile resume so it cannot fork profiles', async () => {
    // A tile opens a session owned by another profile. Resuming without the
    // profile lets the gateway fall back to the launch-profile DB and clone the
    // conversation into the wrong profile (#67603). The owning profile must ride
    // both the transcript prefetch and the resume RPC.
    setSessions([row({ id: 'stored-x', profile: 'ai-engineer' })])

    const requestGateway = vi.fn(async (method: string) =>
      method === 'session.resume' ? ({ session_id: 'runtime-1' } as never) : ({} as never)
    )

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-1' } as never)

    renderTile(requestGateway)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-x')

    expect(runtimeId).toBe('runtime-1')
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-x', 'ai-engineer')
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'ai-engineer',
      'session.resume',
      {
        session_id: 'stored-x',
        cols: 96,
        profile: 'ai-engineer',
        omit_messages: true
      },
      undefined,
      undefined
    )
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('resolves and carries a default-profile session explicitly', async () => {
    setSessions([row({ id: 'stored-y', profile: 'default' })])

    const requestGateway = vi.fn(async () => ({}) as never)

    // #92961: a known owner is ALWAYS routed through the profile router —
    // even 'default' — never dispatched on the ambient socket.
    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-2' } as never)

    renderTile(requestGateway)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-y')

    expect(runtimeId).toBe('runtime-2')
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'default',
      'session.resume',
      {
        session_id: 'stored-y',
        cols: 96,
        profile: 'default',
        omit_messages: true
      },
      undefined,
      undefined
    )
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('recreates a never-persisted tab runtime under the saved draft key after restart', async () => {
    const draftId = '20260825_032826_012255'
    stashSessionDraft(draftId, 'unsent draft survives restart', [])

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        throw new Error('4007 session not found')
      }

      if (method === 'session.create') {
        return {
          info: { model: 'gpt-test' },
          message_count: 0,
          messages: [],
          session_id: 'runtime-restored-draft',
          stored_session_id: draftId
        } as never
      }

      throw new Error(`unexpected ${method}: ${JSON.stringify(params)}`)
    })

    const updateSessionState = vi.fn()

    renderTile(requestGateway, { updateSessionState })

    await expect(sessionTileDelegate()!.resumeTile(draftId)).resolves.toBe('runtime-restored-draft')
    expect(requestGateway).toHaveBeenNthCalledWith(1, 'session.resume', {
      cols: 96,
      omit_messages: true,
      session_id: draftId
    })
    expect(requestGateway).toHaveBeenNthCalledWith(2, 'session.create', {
      cols: 96,
      restore_session_id: draftId,
      source: 'desktop'
    })
    expect(updateSessionState).toHaveBeenCalledWith('runtime-restored-draft', expect.any(Function), draftId)

    clearSessionDraft(draftId)
  })

  it('restores a never-persisted draft through its legacy profile owner', async () => {
    const draftId = '20260825_032826_012255'
    const desktopBridge = window.hermesDesktop

    Reflect.deleteProperty(window, 'hermesDesktop')
    $connectionsRegistry.set(null)
    $profiles.set([{ name: 'default' }, { name: 'ops' }] as never)
    $activeGatewayProfile.set('ops')
    stashSessionDraft(draftId, 'profile-scoped draft survives restart', [])

    vi.mocked(requestGatewayForProfile).mockImplementation(async (_profile, method) => {
      if (method === 'session.resume') {
        throw new Error('4007 session not found')
      }

      if (method === 'session.create') {
        return {
          info: {},
          message_count: 0,
          messages: [],
          session_id: 'runtime-restored-profile-draft',
          stored_session_id: draftId
        } as never
      }

      throw new Error(`unexpected ${method}`)
    })

    const ambientRequest = vi.fn(async () => ({}) as never)

    try {
      renderTile(ambientRequest)

      await expect(sessionTileDelegate()!.resumeTile(draftId)).resolves.toBe('runtime-restored-profile-draft')
      expect(requestGatewayForProfile).toHaveBeenNthCalledWith(
        1,
        'ops',
        'session.resume',
        { cols: 96, omit_messages: true, profile: 'ops', session_id: draftId },
        undefined,
        undefined
      )
      expect(requestGatewayForProfile).toHaveBeenNthCalledWith(
        2,
        'ops',
        'session.create',
        { cols: 96, profile: 'ops', restore_session_id: draftId, source: 'desktop' },
        undefined,
        undefined
      )
      expect(ambientRequest).not.toHaveBeenCalled()
    } finally {
      clearSessionDraft(draftId)
      $profiles.set([])
      $activeGatewayProfile.set('default')
      $connectionsRegistry.set(null)

      if (desktopBridge) {
        window.hermesDesktop = desktopBridge
      }
    }
  })

  it('single-flights the complete resume-or-restore operation for concurrent callers', async () => {
    const draftId = '20260825_032826_012255'
    stashSessionDraft(draftId, 'saved concurrent draft', [])

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        await Promise.resolve()
        throw new Error('4007 session not found')
      }

      if (method === 'session.create') {
        await Promise.resolve()

        return {
          info: {},
          message_count: 0,
          messages: [],
          session_id: 'runtime-shared-restore',
          stored_session_id: draftId
        } as never
      }

      throw new Error(`unexpected ${method}`)
    })

    renderTile(requestGateway)

    const [first, second] = await Promise.all([
      sessionTileDelegate()!.resumeTile(draftId),
      sessionTileDelegate()!.resumeTile(draftId)
    ])

    expect(first).toBe('runtime-shared-restore')
    expect(second).toBe(first)
    expect(requestGateway.mock.calls.filter(([method]) => method === 'session.resume')).toHaveLength(1)
    expect(requestGateway.mock.calls.filter(([method]) => method === 'session.create')).toHaveLength(1)

    clearSessionDraft(draftId)
  })

  it('rejects and closes a fallback runtime when an older backend changes the draft key', async () => {
    const draftId = '20260825_032826_012255'
    stashSessionDraft(draftId, 'saved draft', [])

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('4007 session not found')
      }

      if (method === 'session.create') {
        return {
          info: {},
          message_count: 0,
          messages: [],
          session_id: 'runtime-wrong-key',
          stored_session_id: '20260825_040000_deadbe'
        } as never
      }

      if (method === 'session.close') {
        return {} as never
      }

      throw new Error(`unexpected ${method}`)
    })

    renderTile(requestGateway)

    await expect(sessionTileDelegate()!.resumeTile(draftId)).rejects.toThrow('did not preserve its session key')
    expect(requestGateway).toHaveBeenNthCalledWith(3, 'session.close', { session_id: 'runtime-wrong-key' })

    clearSessionDraft(draftId)
  })

  it('does not fork an ordinary missing session when there is no saved local draft', async () => {
    const missingId = '20260825_030000_abcdef'
    clearSessionDraft(missingId)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('4007 session not found')
      }

      throw new Error(`unexpected ${method}`)
    })

    renderTile(requestGateway)

    await expect(sessionTileDelegate()!.resumeTile(missingId)).rejects.toThrow('4007 session not found')
    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  it('carries a session row connection owner into a same-named tile resume', async () => {
    setSessions([row({ connection_id: 'source-b', id: 'stored-shared', profile: 'default' })])

    const ambientRequest = vi.fn(async () => ({}) as never)
    vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({ session_id: 'runtime-shared' } as never)

    renderTile(ambientRequest)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-shared')

    expect(runtimeId).toBe('runtime-shared')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-b', 'default', 'session.resume', {
      session_id: 'stored-shared',
      cols: 96,
      omit_messages: true,
      profile: 'default'
    })
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('routes a Bot tile prefetch and resume through its exact connection owner', async () => {
    const route = {
      connectionId: 'barry',
      mode: 'remote' as const,
      profile: 'oxcoder',
      targetProfile: 'backend-oxcoder'
    }

    setSessionOwnerHint('stored-remote', route)
    vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({ session_id: 'runtime-remote' } as never)
    const ambientRequest = vi.fn(async () => ({}) as never)

    renderTile(ambientRequest)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-remote')

    expect(runtimeId).toBe('runtime-remote')
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-remote', {
      connectionId: 'barry',
      profile: 'backend-oxcoder'
    })
    expect(requestGatewayForAgent).toHaveBeenCalledWith('barry', 'oxcoder', 'session.resume', {
      session_id: 'stored-remote',
      cols: 96,
      omit_messages: true,
      profile: 'backend-oxcoder'
    })
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('reuses a warm binding that still carries a transcript', async () => {
    const stateA = { busy: false, messages: [{ id: 'm1' }], storedSessionId: 'stored-a' }
    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-a', 'runtime-a']]) }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-a', stateA]]) }
    const requestGateway = vi.fn(async () => ({}) as never)

    renderTile(requestGateway, { runtimeIdByStoredSessionIdRef, sessionStateByRuntimeIdRef })
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-a')

    expect(runtimeId).toBe('runtime-a')
    expect(requestGateway).not.toHaveBeenCalled()
    expect(getLatestSessionMessages).not.toHaveBeenCalled()
  })

  it('merges persisted messages into a warm tile on explicit reopen (#96183)', async () => {
    const stateA = {
      busy: false,
      messages: [{ id: 'm1', parts: [{ type: 'text', text: 'old' }], role: 'user' }],
      storedSessionId: 'stored-a'
    }

    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-a', 'runtime-a']]) }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-a', stateA]]) }
    const updateSessionState = vi.fn((_id, updater) => updater(stateA))
    const requestGateway = vi.fn(async () => ({}) as never)

    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce({
      messages: [
        { id: 'm1', content: 'old', role: 'user' },
        { id: 'm2', content: 'cron delivery', role: 'user' }
      ],
      session_id: 'stored-a'
    } as never)

    renderTile(requestGateway, { runtimeIdByStoredSessionIdRef, sessionStateByRuntimeIdRef, updateSessionState })
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-a', { refreshTranscript: true })

    expect(runtimeId).toBe('runtime-a')
    expect(requestGateway).not.toHaveBeenCalled()
    expect(getLatestSessionMessages).toHaveBeenCalled()
    expect(updateSessionState).toHaveBeenCalled()

    const updater = updateSessionState.mock.calls[0][1] as (state: typeof stateA) => {
      messages: Array<{ parts?: Array<{ text?: string }> }>
    }

    const next = updater(stateA)
    const texts = next.messages.flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))

    expect(texts.some(text => text.includes('cron delivery'))).toBe(true)
  })

  it('falls through to a real resume when the warm binding has no transcript (post-wake empty tile)', async () => {
    // Sleep/wake regression: a released/stale cached state (messages: []) must
    // NOT satisfy the warm path — reusing it re-bound the tile to a dead
    // runtime id and painted the pane permanently empty.
    setSessions([row({ id: 'stored-b', profile: 'default' })])

    const staleState = { busy: false, messages: [], storedSessionId: 'stored-b' }
    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-b', 'runtime-dead']]) }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-dead', staleState]]) }

    const requestGateway = vi.fn(async () => ({}) as never)

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-fresh' } as never)

    renderTile(requestGateway, { runtimeIdByStoredSessionIdRef, sessionStateByRuntimeIdRef })
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-b')

    expect(runtimeId).toBe('runtime-fresh')
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'default',
      'session.resume',
      {
        session_id: 'stored-b',
        cols: 96,
        profile: 'default',
        omit_messages: true
      },
      undefined,
      undefined
    )
  })

  it('hydrates the tile model and provider from resume info', async () => {
    setSessions([row({ id: 'stored-model', profile: 'default' })])

    const updateSessionState = vi.fn()

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({
      info: { fast: true, model: 'gpt-5', provider: 'openai', reasoning_effort: 'high', running: false },
      session_id: 'runtime-model'
    } as never)

    renderTile(vi.fn(), { updateSessionState })
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-model')

    expect(runtimeId).toBe('runtime-model')
    expect(updateSessionState).toHaveBeenCalled()

    const updater = updateSessionState.mock.calls[0][1] as (state: { messages: unknown[] }) => Record<string, unknown>
    const next = updater({ messages: [] })

    expect(next.model).toBe('gpt-5')
    expect(next.provider).toBe('openai')
    expect(next.reasoningEffort).toBe('high')
    expect(next.fast).toBe(true)
  })

  it('invalidateRuntimeBindings clears the stored→runtime map so tiles re-resume after reconnect', async () => {
    setSessions([row({ id: 'stored-c', profile: 'default' })])

    const liveState = { busy: false, messages: [{ id: 'm1' }], storedSessionId: 'stored-c' }
    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-c', 'runtime-dead']]) }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-dead', liveState]]) }

    const requestGateway = vi.fn(async () => ({}) as never)

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-fresh' } as never)

    renderTile(requestGateway, { runtimeIdByStoredSessionIdRef, sessionStateByRuntimeIdRef })

    // Gateway reconnect (what resetTileRuntimeBindings calls on wake):
    sessionTileDelegate()!.invalidateRuntimeBindings!()
    expect(runtimeIdByStoredSessionIdRef.current.size).toBe(0)

    // The next resume goes cold instead of reusing the dead binding.
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-c')
    expect(runtimeId).toBe('runtime-fresh')
  })
})

describe('useSessionTileDelegate retireBusyClaim', () => {
  it('retires a stale busy claim through the session-state write path (#93059)', () => {
    const busyState = { awaitingResponse: true, busy: true, messages: [{ id: 'm1' }], storedSessionId: 'stored-d' }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-dead', busyState]]) }
    const updateSessionState = vi.fn()

    renderTile(
      vi.fn(async () => ({}) as never),
      { sessionStateByRuntimeIdRef, updateSessionState }
    )

    expect(sessionTileDelegate()!.retireBusyClaim!('runtime-dead')).toBe(true)
    expect(updateSessionState).toHaveBeenCalledWith('runtime-dead', expect.any(Function))

    // The updater is the downgrade: busy/awaiting off, everything else intact.
    const updater = updateSessionState.mock.calls[0][1] as (state: typeof busyState) => typeof busyState

    expect(updater(busyState)).toEqual({ ...busyState, awaitingResponse: false, busy: false })
  })

  it('reports a miss instead of minting a cache entry for a runtime it never held', () => {
    // No phantoms: updateSessionState mints a state for any id it is handed,
    // and prune never collects a transcript-less entry — so a miss must not
    // reach the write path; the store retires its own mirror instead.
    const idle = { awaitingResponse: false, busy: false, messages: [{ id: 'm1' }], storedSessionId: 'stored-e' }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-idle', idle]]) }
    const updateSessionState = vi.fn()

    renderTile(
      vi.fn(async () => ({}) as never),
      { sessionStateByRuntimeIdRef, updateSessionState }
    )

    expect(sessionTileDelegate()!.retireBusyClaim!('runtime-unknown')).toBe(false)
    expect(sessionTileDelegate()!.retireBusyClaim!('runtime-idle')).toBe(false)
    expect(updateSessionState).not.toHaveBeenCalled()
  })
})

describe('useSessionTileDelegate interruptSession', () => {
  beforeEach(() => {
    setSessions([])
  })

  afterEach(async () => {
    setSessions([])
    const { clearSessionRecentlyInterrupted } = await import('../../session/hooks/use-prompt-actions/utils')
    clearSessionRecentlyInterrupted()
  })

  it('marks the session recently interrupted so a quick tile edit/resend still interrupt-firsts (#83855)', async () => {
    const { isSessionRecentlyInterrupted } = await import('../../session/hooks/use-prompt-actions/utils')

    const requestGateway = vi.fn(async () => ({}) as never)

    renderTile(requestGateway)
    await sessionTileDelegate()!.interruptSession('runtime-tile-1')

    expect(requestGateway).toHaveBeenCalledWith('session.interrupt', { session_id: 'runtime-tile-1' })
    // Same 3s cooldown the primary chat's Stop sets: busy reads false while the
    // gateway winds down, so the rewind path must still interrupt-first.
    expect(isSessionRecentlyInterrupted('runtime-tile-1')).toBe(true)
  })
})
