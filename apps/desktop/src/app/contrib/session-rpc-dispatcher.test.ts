import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Fail-closed owner resolution for the window's ONE session-scoped RPC
// dispatcher. A request that names a session whose owner NO rung can name
// (tile route → exact hint → connection-tagged / profiled row → REST probe)
// must not ride the ambient presentation socket: "active" has no routing
// authority, and the fallback turned missing ownership metadata into a
// misleading backend "session not found". The single exception is the legacy
// single-backend Desktop (no registry source, ≤1 profile), where the ambient
// gateway IS the owner by construction.

const gatewayMocks = vi.hoisted(() => ({
  activeConnectionId: null as null | string,
  requestGatewayForAgent: vi.fn(async () => ({ routed: true })),
  requestGatewayForProfile: vi.fn(async () => ({ profiled: true }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  activeGatewayConnectionId: () => gatewayMocks.activeConnectionId,
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent,
  requestGatewayForProfile: gatewayMocks.requestGatewayForProfile
}))

const probe = vi.hoisted(() => ({ resolveSessionOwner: vi.fn(async () => undefined as unknown) }))
const sessionMocks = vi.hoisted(() => ({ requestSessionResume: vi.fn() }))

const storedTranscript = vi.hoisted(() => ({
  fetchStoredTranscriptAcrossBackends: vi.fn(async () => null as unknown)
}))

vi.mock('@/app/session/hooks/use-session-actions/utils', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  resolveSessionOwner: probe.resolveSessionOwner
}))

vi.mock('@/api/sessions', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  fetchStoredTranscriptAcrossBackends: storedTranscript.fetchStoredTranscriptAcrossBackends
}))

vi.mock('@/store/session', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestSessionResume: sessionMocks.requestSessionResume
}))

const { createSessionRpcDispatcher, readOnlyResumeResponse } = await import('./session-rpc-dispatcher')
const { $connectionsRegistry } = await import('@/store/connection-registry-state')
const { $profiles } = await import('@/store/profile')

const { _resetSessionOwnerHintsForTests, setCronSessions, setMessagingSessions, setSessionOwnerHint, setSessions } =
  await import('@/store/session')

const { isSessionOwnerResolutionError } = await import('@/store/session-owner-resolution')

const { $readOnlyStoredTranscripts, isReadOnlyRuntimeId, isStoredTranscriptReadOnly } =
  await import('@/store/read-only-transcript')

const { $removedSessionIds, $sessionMutationsInFlight } = await import('@/store/session-removal')
const { $sessionTiles, recordSessionEventScope } = await import('@/store/session-states')
const { makeSessionInfo } = await import('@/test/session-info')

function dispatcher(
  ambientRequest = vi.fn(async () => ({ ambient: true })),
  selectedStoredSessionId: null | string = null
) {
  return {
    ambientRequest,
    request: createSessionRpcDispatcher({
      ambientRequest: ambientRequest as never,
      runtimeIdByStoredSessionIdRef: { current: new Map([['stored-omar', 'rt-omar']]) },
      selectedStoredSessionIdRef: { current: selectedStoredSessionId },
      sessionStateByRuntimeIdRef: { current: new Map() }
    })
  }
}

beforeEach(() => {
  gatewayMocks.activeConnectionId = 'local'
  $connectionsRegistry.set({ connections: [{ id: 'local' }] } as never)
  $profiles.set([{ name: 'default' }, { name: 'omar' }] as never)
  probe.resolveSessionOwner.mockResolvedValue(undefined)
  storedTranscript.fetchStoredTranscriptAcrossBackends.mockResolvedValue(null)
  $readOnlyStoredTranscripts.set(new Set())
})

afterEach(() => {
  $connectionsRegistry.set(null)
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  $sessionTiles.set([])
  $profiles.set([])
  $removedSessionIds.set(new Set())
  $sessionMutationsInFlight.set(new Set())
  $readOnlyStoredTranscripts.set(new Set())
  _resetSessionOwnerHintsForTests({ storage: true })
  sessionMocks.requestSessionResume.mockReset()
  vi.clearAllMocks()
})

describe('createSessionRpcDispatcher: fail closed', () => {
  it('rejects with an explicit owner-resolution error instead of riding the ambient socket', async () => {
    const { ambientRequest, request } = dispatcher()

    await expect(request('prompt.submit', { session_id: 'rt-orphan', text: 'hi' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    await expect(request('prompt.submit', { session_id: 'rt-orphan', text: 'hi' })).rejects.toThrow(
      /owner could not be resolved for "rt-orphan" \(prompt.submit\)/
    )

    expect(probe.resolveSessionOwner).toHaveBeenCalledWith('rt-orphan')
    expect(ambientRequest).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForAgent).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForProfile).not.toHaveBeenCalled()
  })

  it('still lets a request with NO session (ambient chrome) reach the ambient socket', async () => {
    const { ambientRequest, request } = dispatcher()

    await expect(request('config.get', {})).resolves.toEqual({ ambient: true })
    expect(ambientRequest).toHaveBeenCalledWith('config.get', {})
  })

  it('keeps the legacy single-backend Desktop on the ambient socket: no registry source, one profile', async () => {
    gatewayMocks.activeConnectionId = null
    $connectionsRegistry.set(null)
    $profiles.set([{ name: 'default' }] as never)
    const { ambientRequest, request } = dispatcher()

    await expect(request('session.resume', { session_id: 'stored-legacy' })).resolves.toEqual({ ambient: true })
    expect(ambientRequest).toHaveBeenCalledWith('session.resume', { session_id: 'stored-legacy' })
  })

  it('fails closed as soon as there is somewhere to misroute to: a second profile, or a live registry source', async () => {
    gatewayMocks.activeConnectionId = null
    $connectionsRegistry.set(null)
    $profiles.set([{ name: 'default' }, { name: 'omar' }] as never)
    await expect(dispatcher().request('session.resume', { session_id: 'stored-x' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )

    gatewayMocks.activeConnectionId = 'local'
    $connectionsRegistry.set({ connections: [{ id: 'local' }] } as never)
    $profiles.set([{ name: 'default' }] as never)
    await expect(dispatcher().request('session.resume', { session_id: 'stored-x' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
  })
})

describe('createSessionRpcDispatcher: session.resume no-owner recovery (#102618)', () => {
  // #94724 gave the tile-delegate path a read-only stored-transcript recovery,
  // but this dispatcher kept the BARE fail-closed gate — so an orphaned /
  // owner-less session whose resume surfaced here dead-ended on "Couldn't open
  // this session" behind a Retry that re-ran the same resume forever. The
  // recovery has to live on both doors.
  const transcript = {
    messages: [
      { content: 'what did we decide?', role: 'user' },
      { content: 'we shipped it', role: 'assistant' }
    ],
    session_id: 'stored-orphan'
  }

  it('answers from the stored transcript instead of throwing, and never routes a live session', async () => {
    storedTranscript.fetchStoredTranscriptAcrossBackends.mockResolvedValue(transcript)
    const { ambientRequest, request } = dispatcher()

    const resumed = await request<{ message_count: number; messages: unknown[]; session_id: string }>(
      'session.resume',
      { session_id: 'stored-orphan' }
    )

    expect(storedTranscript.fetchStoredTranscriptAcrossBackends).toHaveBeenCalledWith('stored-orphan')
    expect(isReadOnlyRuntimeId(resumed.session_id)).toBe(true)
    expect(resumed.messages).toEqual(transcript.messages)
    expect(resumed.message_count).toBe(2)

    // The whole point: no live routing happened on any socket.
    expect(ambientRequest).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForAgent).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForProfile).not.toHaveBeenCalled()

    // Latched so composer/submit surfaces refuse writes into a session with
    // no routable runtime.
    expect(isStoredTranscriptReadOnly('stored-orphan')).toBe(true)
  })

  it('still rejects with the ORIGINAL owner error when no backend holds the transcript', async () => {
    storedTranscript.fetchStoredTranscriptAcrossBackends.mockResolvedValue(null)

    await expect(dispatcher().request('session.resume', { session_id: 'stored-nowhere' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    expect(isStoredTranscriptReadOnly('stored-nowhere')).toBe(false)
  })

  it('leaves every OTHER session-scoped method failing closed', async () => {
    storedTranscript.fetchStoredTranscriptAcrossBackends.mockResolvedValue(transcript)

    await expect(dispatcher().request('prompt.submit', { session_id: 'stored-orphan', text: 'hi' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    await expect(dispatcher().request('session.activate', { session_id: 'stored-orphan' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    expect(storedTranscript.fetchStoredTranscriptAcrossBackends).not.toHaveBeenCalled()
  })

  it('does not recover a resume that failed for a REAL reason once the owner is known', async () => {
    // A resolvable owner never enters the recovery path: a transport failure
    // must keep its own semantics rather than being papered over with history.
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    storedTranscript.fetchStoredTranscriptAcrossBackends.mockResolvedValue(transcript)
    gatewayMocks.requestGatewayForAgent.mockRejectedValueOnce(new Error('backend exploded'))

    await expect(dispatcher().request('session.resume', { session_id: 'stored-omar' })).rejects.toThrow(
      'backend exploded'
    )
    expect(storedTranscript.fetchStoredTranscriptAcrossBackends).not.toHaveBeenCalled()
  })

  it('clears a stale read-only latch once the owner resolves again', async () => {
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    // Round 1: unresolvable owner (no row) → read-only.
    storedTranscript.fetchStoredTranscriptAcrossBackends.mockResolvedValue(transcript)
    setSessions([])
    await dispatcher().request('session.resume', { session_id: 'stored-omar' })
    expect(isStoredTranscriptReadOnly('stored-omar')).toBe(true)

    // Round 2: the row is back (owner backfilled) → live resume clears the latch.
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    await expect(dispatcher().request('session.resume', { session_id: 'stored-omar' })).resolves.toEqual({
      routed: true
    })
    expect(isStoredTranscriptReadOnly('stored-omar')).toBe(false)
  })

  it('readOnlyResumeResponse reports an idle, non-running snapshot', () => {
    const response = readOnlyResumeResponse('stored-9', transcript.messages as never)

    expect(response.running).toBe(false)
    expect(response.resumed).toBe('stored-9')
    expect(isReadOnlyRuntimeId(response.session_id)).toBe(true)
  })
})


describe('createSessionRpcDispatcher: exact owner rungs', () => {
  it('routes by the connection-tagged row when the hint is gone (runtime id translated to the stored id)', async () => {
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    const { ambientRequest, request } = dispatcher()

    await expect(request('prompt.submit', { session_id: 'rt-omar', text: 'again' })).resolves.toEqual({ routed: true })

    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledWith('local', 'omar', 'prompt.submit', {
      session_id: 'rt-omar',
      text: 'again'
    })
    expect(ambientRequest).not.toHaveBeenCalled()
    expect(probe.resolveSessionOwner).not.toHaveBeenCalled()
  })

  it("routes a freshly recovered runtime id to its session's owner only once the binding is published", async () => {
    // A session-scoped retry after withSessionNotFoundResume: the new runtime id is
    // known to nobody until the recovering caller publishes stored → runtime. Before
    // that, the dispatcher cannot translate it and fails closed; after, it reaches the
    // stored session's owner. Callers therefore publish in `onRecovered`, which runs
    // before the retried call — not after the retry has already returned.
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-omar', 'rt-omar-dead']]) }
    const ambientRequest = vi.fn(async () => ({ ambient: true }))

    const request = createSessionRpcDispatcher({
      ambientRequest: ambientRequest as never,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionIdRef: { current: null },
      sessionStateByRuntimeIdRef: { current: new Map() }
    })

    await expect(request('image.attach', { path: '/tmp/shot.png', session_id: 'rt-omar-live' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    expect(gatewayMocks.requestGatewayForAgent).not.toHaveBeenCalled()

    runtimeIdByStoredSessionIdRef.current.set('stored-omar', 'rt-omar-live')

    await expect(request('image.attach', { path: '/tmp/shot.png', session_id: 'rt-omar-live' })).resolves.toEqual({
      routed: true
    })
    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledWith('local', 'omar', 'image.attach', {
      path: '/tmp/shot.png',
      session_id: 'rt-omar-live'
    })
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('prefers the exact hint over an untagged row profile, and the probe result over nothing', async () => {
    setSessions([makeSessionInfo({ id: 'stored-omar', profile: 'default' })])
    setSessionOwnerHint('stored-omar', { connectionId: 'local', profile: 'omar' })

    await expect(dispatcher().request('session.interrupt', { session_id: 'rt-omar' })).resolves.toEqual({
      routed: true
    })
    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenLastCalledWith('local', 'omar', 'session.interrupt', {
      session_id: 'rt-omar'
    })

    _resetSessionOwnerHintsForTests()
    setSessions([])
    probe.resolveSessionOwner.mockResolvedValue({ connectionId: 'homelab', profile: 'worker' })

    await expect(dispatcher().request('session.activate', { session_id: 'stored-hidden' })).resolves.toEqual({
      routed: true
    })
    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenLastCalledWith('homelab', 'worker', 'session.activate', {
      session_id: 'stored-hidden'
    })
  })

  it('resolves owners from the cron and messaging sidebar slices, not just recents (cron approval.respond)', async () => {
    // A scheduler-minted cron session has no tile, no hint, and no row in
    // $sessions — its row lives in the sidebar's cron slice. The row rung must
    // see that slice, or the approval raised inside a cron chat fails closed
    // with SessionOwnerResolutionError and can never be answered.
    setCronSessions([makeSessionInfo({ id: 'stored-cron', profile: 'omar', source: 'cron' })])
    const { ambientRequest, request } = dispatcher()

    await expect(
      request('approval.respond', { choice: 'once', request_id: 'req-1', session_id: 'stored-cron' })
    ).resolves.toEqual({ profiled: true })
    expect(gatewayMocks.requestGatewayForProfile).toHaveBeenLastCalledWith(
      'omar',
      'approval.respond',
      {
        choice: 'once',
        request_id: 'req-1',
        session_id: 'stored-cron'
      },
      undefined,
      undefined
    )
    expect(ambientRequest).not.toHaveBeenCalled()
    expect(probe.resolveSessionOwner).not.toHaveBeenCalled()

    // Messaging slice, connection-tagged row → exact route.
    setMessagingSessions([makeSessionInfo({ connection_id: 'homelab', id: 'stored-tg', profile: 'bots' })])

    await expect(request('prompt.submit', { session_id: 'stored-tg', text: 'hi' })).resolves.toEqual({ routed: true })
    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenLastCalledWith('homelab', 'bots', 'prompt.submit', {
      session_id: 'stored-tg',
      text: 'hi'
    })
  })
})

describe('createSessionRpcDispatcher: routes by the session OWNING connection when two connections share a profile name', () => {
  // Two registered connections, both exposing `default` (the default install:
  // this device + a remote gateway), and the window's primary is the OTHER
  // connection. A bare profile name carries no connection identity, and the
  // profile door (requestGatewayForProfile -> gatewayForProfile) resolves a
  // bare name equal to the primary profile against the PRIMARY socket — a
  // different machine than the one holding the session. The inbound event
  // already proved which socket owns the runtime, so that exact owner has to
  // outrank the connection-blind row profile: otherwise the 2nd prompt of a
  // `This device` chat is answered by the remote backend with
  // `4001 session not found`.
  function twoConnections(): void {
    $connectionsRegistry.set({ connections: [{ id: 'local' }, { id: 'homelab' }] } as never)
  }

  it('routes a session that lives on the LOCAL connection back to that connection', async () => {
    gatewayMocks.activeConnectionId = 'homelab'
    twoConnections()
    recordSessionEventScope({ connectionId: 'local', profile: 'default', session_id: 'rt-local' })
    setSessions([makeSessionInfo({ id: 'rt-local', profile: 'default' })])
    const { ambientRequest, request } = dispatcher()

    await expect(request('prompt.submit', { session_id: 'rt-local', text: 'again' })).resolves.toEqual({
      routed: true
    })

    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledWith('local', 'default', 'prompt.submit', {
      session_id: 'rt-local',
      text: 'again'
    })
    expect(gatewayMocks.requestGatewayForProfile).not.toHaveBeenCalled()
    expect(ambientRequest).not.toHaveBeenCalled()
  })
})

describe('createSessionRpcDispatcher: stale runtime recovery', () => {
  it('requests a durable rebind for the visible session after a structured 4001', async () => {
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    gatewayMocks.requestGatewayForAgent.mockRejectedValueOnce(
      Object.assign(new Error('runtime was reaped'), { code: 4001 })
    )
    const { request } = dispatcher(undefined, 'stored-omar')

    await expect(request('process.list', { session_id: 'rt-omar' })).rejects.toThrow('runtime was reaped')

    expect(sessionMocks.requestSessionResume).toHaveBeenCalledWith('stored-omar', {
      connectionId: 'local',
      profile: 'omar'
    })
  })

  it('does not let a background 4001 pull a different session into the foreground', async () => {
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    gatewayMocks.requestGatewayForAgent.mockRejectedValueOnce(
      Object.assign(new Error('session not found'), { code: 4001 })
    )
    const { request } = dispatcher(undefined, 'stored-other')

    await expect(request('process.list', { session_id: 'rt-omar' })).rejects.toThrow('session not found')

    expect(sessionMocks.requestSessionResume).not.toHaveBeenCalled()
  })

  it('does not interpret an unrelated coded RPC failure as a stale runtime', async () => {
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    gatewayMocks.requestGatewayForAgent.mockRejectedValueOnce(
      Object.assign(new Error('tool output says session not found'), { code: 5007 })
    )
    const { request } = dispatcher(undefined, 'stored-omar')

    await expect(request('process.list', { session_id: 'rt-omar' })).rejects.toThrow(
      'tool output says session not found'
    )

    expect(sessionMocks.requestSessionResume).not.toHaveBeenCalled()
  })

  it('leaves the warm resume lifecycle to recover its own session.activate failure', async () => {
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-omar', profile: 'omar' })])
    gatewayMocks.requestGatewayForAgent.mockRejectedValueOnce(
      Object.assign(new Error('session not found'), { code: 4001 })
    )
    const { request } = dispatcher(undefined, 'stored-omar')

    await expect(request('session.activate', { session_id: 'rt-omar' })).rejects.toThrow('session not found')

    expect(sessionMocks.requestSessionResume).not.toHaveBeenCalled()
  })
})
