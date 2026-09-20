import { beforeEach, describe, expect, it, vi } from 'vitest'

// `host.submitToSession` composes the store's own primitives, so the sequence
// under test is the relation between them: which socket each RPC rides, which id
// each one carries, and who releases which hold.
const mocks = vi.hoisted(() => {
  const releaseRoute = vi.fn()
  const releaseTurn = vi.fn()

  return {
    releaseRoute,
    releaseTurn,
    requestGatewayForAgent: vi.fn(),
    retainGatewayForAgent: vi.fn(async () => releaseRoute),
    retainGatewayForSessionTurn: vi.fn(async () => releaseTurn)
  }
})

vi.mock('@/app/chat/session-view', async () => {
  const { atom } = await import('nanostores')

  return { PRIMARY_SESSION_VIEW: { $awaitingResponse: atom(false), $busy: atom(false) } }
})
vi.mock('@/app/open-session', () => ({ openSession: vi.fn() }))
vi.mock('@/components/pane-shell/tree/store', async () => {
  const { atom } = await import('nanostores')

  return { $narrowViewport: atom(false) }
})
vi.mock('@/contrib/events', () => ({ onGatewayEvent: vi.fn() }))
vi.mock('@/hermes', () => ({ deleteProfile: vi.fn(), getLogs: vi.fn(), getStatus: vi.fn(), hermesApi: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/system-actions', () => ({ runGatewayRestart: vi.fn() }))
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  type LineageRow = { _lineage_root_id?: null | string; id: string }

  return {
    $activeSessionId: atom(null),
    $connection: atom(null),
    $cronSessions: atom([]),
    $currentCwd: atom(''),
    $currentModel: atom(''),
    $gatewayState: atom('open'),
    $messages: atom([]),
    $messagingSessions: atom([]),
    $selectedStoredSessionId: atom(null),
    $sessions: atom([]),
    $unreadFinishedSessionIds: atom([]),
    getSessionOwnerHints: vi.fn(() => ({})),
    lineageAliases: (storedId: string) => [storedId],
    rememberedSessionProfile: (_sessions: unknown, _id: null | string, activeProfile: null | string) =>
      (activeProfile ?? '').trim() || 'default',
    requestSessionResume: vi.fn(),
    sessionMatchesStoredId: (session: LineageRow, storedSessionId: string) =>
      session.id === storedSessionId || session._lineage_root_id === storedSessionId,
    sessionPinId: (session: LineageRow) => session._lineage_root_id ?? session.id,
    setResumeExhaustedSessionId: vi.fn(),
    setSessionOwnerHint: vi.fn()
  }
})
vi.mock('@/store/session-states', async () => {
  const { atom } = await import('nanostores')

  return {
    $attentionSessionIds: atom([]),
    $draftSessionIds: atom([]),
    $focusedRuntimeId: atom(null),
    $focusedSessionState: atom(null),
    $focusedStoredSessionId: atom(null),
    $sessionStates: atom({}),
    $sessionTiles: atom([]),
    $stalledSessionIds: atom([]),
    $workingSessionIds: atom([]),
    dropTilesForProfile: vi.fn(),
    focusWorkspaceOwnerSessionTile: vi.fn(),
    sessionTileDelegate: vi.fn(() => null)
  }
})
vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  const profiles = atom([])

  return {
    $activeGatewayProfile: atom('default'),
    $gatewaySwapTarget: atom(null),
    $hydrationSyncProfile: atom(null),
    $profiles: profiles,
    $showAllProfiles: atom(false),
    ensureGatewayAgent: vi.fn(),
    ensureGatewayProfile: vi.fn(),
    newSessionInAgent: vi.fn(),
    newSessionInProfile: vi.fn(),
    normalizeProfileKey: (value: null | string | undefined) => (value ?? '').trim() || 'default',
    prewarmProfileBackend: vi.fn(),
    refreshProfiles: vi.fn(async () => profiles.get()),
    selectProfile: vi.fn(),
    setActiveProfile: vi.fn(),
    setShowAllProfiles: vi.fn()
  }
})
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayRoute: atom('default'),
    $gateway: atom(null),
    activeGateway: vi.fn(() => null),
    activeGatewayConnectionId: vi.fn(() => 'local'),
    ensureGatewayForAgent: vi.fn(),
    openGatewayForAgent: vi.fn(),
    openGatewayForProfile: vi.fn(),
    requestGatewayForAgent: mocks.requestGatewayForAgent,
    requestGatewayForProfile: vi.fn(),
    retainGatewayForAgent: mocks.retainGatewayForAgent,
    retainGatewayForRelay: vi.fn(() => () => undefined),
    retainGatewayForSessionTurn: mocks.retainGatewayForSessionTurn,
    retireLocalProfileGateways: vi.fn()
  }
})

const { host } = await import('./index')

/** A remote route where the Desktop-side profile name and the backend profile
 *  differ — the shape that makes an ambient-profile fallback visibly wrong. */
const route = {
  connectionId: 'source-b',
  mode: 'remote' as const,
  profile: 'remote-worker',
  targetProfile: 'backend-worker'
}

const STORED_ID = '20260920_122038_661ec0'

beforeEach(() => {
  mocks.releaseRoute.mockClear()
  mocks.releaseTurn.mockClear()
  mocks.requestGatewayForAgent.mockReset()
  mocks.retainGatewayForAgent.mockReset()
  mocks.retainGatewayForAgent.mockImplementation(async () => mocks.releaseRoute)
  mocks.retainGatewayForSessionTurn.mockReset()
  mocks.retainGatewayForSessionTurn.mockImplementation(async () => mocks.releaseTurn)
})

describe('host.submitToSession', () => {
  it('resumes the stored session on its own route, then queues the turn against the runtime id', async () => {
    mocks.requestGatewayForAgent.mockImplementation(async (_connectionId, _profile, method) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-9', session_key: STORED_ID }
      }

      if (method === 'prompt.submit') {
        return { status: 'queued' }
      }

      throw new Error(`unexpected method ${method}`)
    })

    const result = await host.submitToSession(route, { storedSessionId: STORED_ID, text: 'pick up the next task' })

    // The route hold exists before the first session-scoped RPC, or the socket
    // is reaped between resume and submit and the runtime goes with it.
    expect(mocks.retainGatewayForAgent).toHaveBeenCalledWith('source-b', 'remote-worker')
    expect(mocks.retainGatewayForAgent.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.requestGatewayForAgent.mock.invocationCallOrder[0]
    )

    const [resumeCall, submitCall] = mocks.requestGatewayForAgent.mock.calls

    // Connection-qualified route, STORED id in, route's backend profile stamped.
    expect(resumeCall.slice(0, 3)).toEqual(['source-b', 'remote-worker', 'session.resume'])
    expect(resumeCall[3]).toEqual({
      session_id: STORED_ID,
      source: 'desktop',
      omit_messages: true,
      profile: 'backend-worker'
    })

    // Addressed by the RUNTIME id resume returned, and fixed to the queue: a
    // background delivery must never steer or redirect a live turn.
    expect(submitCall.slice(0, 3)).toEqual(['source-b', 'remote-worker', 'prompt.submit'])
    expect(submitCall[3]).toEqual({ session_id: 'runtime-9', text: 'pick up the next task', queued: true })

    // Taken after resume and before submit; the terminal-event path owns its
    // release, so the SDK must not release it here.
    expect(mocks.retainGatewayForSessionTurn).toHaveBeenCalledWith('source-b', 'remote-worker', 'runtime-9')
    expect(mocks.retainGatewayForSessionTurn.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.requestGatewayForAgent.mock.invocationCallOrder[1]
    )
    expect(mocks.releaseTurn).not.toHaveBeenCalled()

    expect(result).toEqual({ runtimeSessionId: 'runtime-9', status: 'queued' })
    expect(mocks.releaseRoute).toHaveBeenCalledOnce()
  })

  it('refuses invalid input before dialing and never leaks a hold or a stray submit on failure', async () => {
    // (a) Boundary validation happens before any route is retained.
    await expect(host.submitToSession(route, { storedSessionId: STORED_ID, text: '   ' })).rejects.toThrow(
      'non-empty text'
    )
    await expect(host.submitToSession(route, { storedSessionId: '', text: 'hello' })).rejects.toThrow(
      'stored session id'
    )
    await expect(
      host.submitToSession({ connectionId: 'source-b', mode: 'remote', profile: '', targetProfile: 'backend-worker' }, {
        storedSessionId: STORED_ID,
        text: 'hello'
      })
    ).rejects.toThrow('Profile route must include connectionId, profile, and targetProfile')

    expect(mocks.retainGatewayForAgent).not.toHaveBeenCalled()
    expect(mocks.requestGatewayForAgent).not.toHaveBeenCalled()

    // (b) A resume that fails never reaches prompt.submit and takes no turn hold.
    mocks.requestGatewayForAgent.mockRejectedValueOnce(new Error('session not found'))

    await expect(host.submitToSession(route, { storedSessionId: 'stored-gone', text: 'hello' })).rejects.toThrow(
      'session not found'
    )

    expect(mocks.requestGatewayForAgent).toHaveBeenCalledTimes(1)
    expect(mocks.retainGatewayForSessionTurn).not.toHaveBeenCalled()
    expect(mocks.releaseRoute).toHaveBeenCalledOnce()

    // (c) A submit that fails releases the turn hold it just took — and a
    // timeout is NOT retried, because the turn may already have started.
    mocks.requestGatewayForAgent
      .mockResolvedValueOnce({ session_id: 'runtime-3' })
      .mockRejectedValueOnce(new Error('gateway timed out'))

    await expect(host.submitToSession(route, { storedSessionId: STORED_ID, text: 'hello' })).rejects.toThrow(
      'gateway timed out'
    )

    expect(mocks.requestGatewayForAgent).toHaveBeenCalledTimes(3)
    expect(mocks.releaseTurn).toHaveBeenCalledOnce()
    expect(mocks.releaseRoute).toHaveBeenCalledTimes(2)
  })
})
