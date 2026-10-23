import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { errorRecoveryPlan } from '@/lib/error-surface'
import { requestGatewayForAgent } from '@/store/gateway'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $sessionStates,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  LIVE_TURN_PROBE_TIMEOUT_MS,
  liveTurnVerdict,
  noteSessionEvent,
  publishSessionState,
  recordSessionEventScope,
  SESSION_WATCHDOG_TIMEOUT_MS,
  type SessionTileDelegate,
  setLiveTurnBackend,
  setSessionTileDelegate
} from './session-states'

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

// Read from the store rather than restated here: these assert what happens on
// either side of the threshold, not what the threshold is.
const WATCHDOG_MS = SESSION_WATCHDOG_TIMEOUT_MS

function state(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState(null), storedSessionId: 's1', ...over }
}

describe('session watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  it('marks a silent session stalled without pretending it finished', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))

    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toContain('s1')
    expect($stalledSessionIds.get()).toContain('s1')
  })

  it('clears stalled on new activity and rearms the watchdog', () => {
    const working = state({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt2', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s2')

    publishSessionState('rt2', { ...working, awaitingResponse: true })
    expect($stalledSessionIds.get()).not.toContain('s2')

    vi.advanceTimersByTime(WATCHDOG_MS - 1)
    expect($stalledSessionIds.get()).not.toContain('s2')
    expect($workingSessionIds.get()).toContain('s2')
  })

  it('clears both running and stalled on an authoritative terminal transition', () => {
    const working = state({ busy: true, storedSessionId: 's3' })
    publishSessionState('rt3', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s3')

    publishSessionState('rt3', { ...working, busy: false })

    expect($workingSessionIds.get()).not.toContain('s3')
    expect($stalledSessionIds.get()).not.toContain('s3')
  })

  it('never marks a session stalled when it settles before the window', () => {
    const working = state({ busy: true, storedSessionId: 's4' })
    publishSessionState('rt4', working)
    publishSessionState('rt4', { ...working, busy: false })
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).not.toContain('s4')
    expect($stalledSessionIds.get()).not.toContain('s4')
  })

  it('clears stalled state and disarms timers on a gateway wipe', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toEqual(['s1'])

    clearAllSessionStates()
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toEqual([])
    expect($stalledSessionIds.get()).toEqual([])
  })
})

describe('computed $workingSessionIds', () => {
  beforeEach(() => {
    clearAllSessionStates()
  })

  afterEach(() => {
    clearAllSessionStates()
  })

  it('reflects busy sessions under the id their surfaces key on', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    publishSessionState('rt2', state({ busy: false, storedSessionId: 's2' }))
    // Not yet persisted, so the runtime id is the only id it has — and the one
    // the row is keyed by until the backend hands a stored id back.
    publishSessionState('rt3', state({ busy: true, storedSessionId: null }))

    expect($workingSessionIds.get()).toEqual(['s1', 'rt3'])
  })

  it('updates when session state changes', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual(['s1'])

    publishSessionState('rt1', state({ busy: false, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual([])
  })
})

const SILENCE_MS = LIVE_TURN_EVENT_SILENCE_MS

function partial(text: string, over: Partial<ClientSessionState> = {}): ClientSessionState {
  return state({
    awaitingResponse: true,
    busy: true,
    messages: [
      {
        id: 'a1',
        parts: [{ type: 'text', text }],
        pending: true,
        role: 'assistant'
      }
    ],
    model: 'any-model',
    sawAssistantPayload: true,
    streamId: 'a1',
    turnLive: true,
    turnStartedAt: Date.now(),
    ...over
  })
}

describe('live turn event silence', () => {
  let releaseBackend: () => void = () => undefined

  // What `session.active_list` answers for the runtime under test.
  const listing = (id: string, status: string) => ({ sessions: [{ id, session_key: `stored-${id}`, status }] })

  function backend(
    answer: (method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<unknown>,
    refreshTranscript?: (runtimeId: string, storedSessionId: string) => Promise<unknown> | unknown
  ) {
    const request = vi.fn(answer)
    releaseBackend = setLiveTurnBackend({ request: request as never, refreshTranscript })

    return request
  }

  const card = (runtimeId: string) => $sessionStates.get()[runtimeId]?.messages.find(message => message.errorSurface)

  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  afterEach(() => {
    releaseBackend()
    releaseBackend = () => undefined
    vi.mocked(requestGatewayForAgent).mockReset()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  it('keeps a silent turn when nothing can answer for it', async () => {
    publishSessionState('rt-alone', partial('long tool call', { storedSessionId: 's-alone' }))
    noteSessionEvent('rt-alone')

    await vi.advanceTimersByTimeAsync(SILENCE_MS * 4)

    expect($workingSessionIds.get()).toContain('s-alone')
    expect($sessionStates.get()['rt-alone']?.interrupted).toBeFalsy()
    expect(card('rt-alone')).toBeUndefined()
  })

  it.each(['working', 'starting'])('keeps a turn the backend reports %s, and asks again next window', async status => {
    publishSessionState('rt-quiet', partial('long tool call', { storedSessionId: 's-quiet' }))
    const request = backend(async () => listing('rt-quiet', status))
    noteSessionEvent('rt-quiet')

    await vi.advanceTimersByTimeAsync(SILENCE_MS * 3)

    expect(request).toHaveBeenCalledTimes(3)
    expect(request).toHaveBeenCalledWith('session.active_list', {}, LIVE_TURN_PROBE_TIMEOUT_MS)
    expect($workingSessionIds.get()).toContain('s-quiet')
    expect(card('rt-quiet')).toBeUndefined()
  })

  it('keeps the turn when the check fails, and asks again next window', async () => {
    publishSessionState('rt-down', partial('partial', { storedSessionId: 's-down' }))

    const request = backend(async () => {
      throw new Error('Hermes gateway unavailable')
    })

    noteSessionEvent('rt-down')

    await vi.advanceTimersByTimeAsync(SILENCE_MS * 2)

    expect(request).toHaveBeenCalledTimes(2)
    expect($workingSessionIds.get()).toContain('s-down')
    expect($sessionStates.get()['rt-down']?.interrupted).toBeFalsy()
    expect(card('rt-down')).toBeUndefined()
  })

  it('does not act on an answer that lands after the turn produced an event', async () => {
    publishSessionState('rt-late', partial('partial', { storedSessionId: 's-late' }))
    let answer!: (value: unknown) => void
    backend(() => new Promise(resolve => (answer = resolve)))
    noteSessionEvent('rt-late')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)
    noteSessionEvent('rt-late')
    answer({ sessions: [] })
    await vi.advanceTimersByTimeAsync(0)

    expect($workingSessionIds.get()).toContain('s-late')
    expect(card('rt-late')).toBeUndefined()
  })

  it('settles a turn the backend reports over, keeping its reply and a live path for late events', async () => {
    // On screen, so the settled transcript stays in the mirror to inspect.
    $activeSessionId.set('rt-done')
    publishSessionState('rt-done', partial('the answer', { storedSessionId: 's-done' }))
    backend(async () => listing('rt-done', 'idle'))
    noteSessionEvent('rt-done')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const settled = $sessionStates.get()['rt-done']
    expect($workingSessionIds.get()).not.toContain('s-done')
    expect(settled?.busy).toBe(false)
    expect(settled?.turnLive).toBe(false)
    // Not a Stop: the rest of the turn's events are still accepted.
    expect(settled?.interrupted).toBeFalsy()
    expect(settled?.heartbeatSettledStreamId).toBe('a1')
    expect(settled?.messages.every(message => !message.pending)).toBe(true)
    expect(chatMessageText(settled!.messages[0])).toBe('the answer')
    expect(card('rt-done')).toBeUndefined()
  })

  it('offers a retry only once the backend confirms a turn ended without a reply', async () => {
    $activeSessionId.set('rt-empty')
    publishSessionState(
      'rt-empty',
      state({ awaitingResponse: true, busy: true, storedSessionId: 's-empty', turnLive: true })
    )
    // A reaped runtime is absent from the list.
    backend(async () => ({ sessions: [] }))
    noteSessionEvent('rt-empty')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const failed = card('rt-empty')
    expect($workingSessionIds.get()).not.toContain('s-empty')
    expect(failed?.errorSurface?.code).toBe('no_reply')
    expect(errorRecoveryPlan(failed?.errorSurface).retry).toBe(true)
    expect(failed?.error).not.toMatch(/connection/i)
  })

  it('pulls the stored reply for a turn on screen before deciding it had none', async () => {
    $activeSessionId.set('rt-screen')
    publishSessionState(
      'rt-screen',
      state({ awaitingResponse: true, busy: true, storedSessionId: 's-screen', turnLive: true })
    )

    const refreshTranscript = vi.fn(async (runtimeId: string) => {
      const current = $sessionStates.get()[runtimeId]!

      publishSessionState(runtimeId, {
        ...current,
        messages: [{ id: 'stored-reply', parts: [textPart('finished on the backend')], role: 'assistant' }]
      })
    })

    backend(async () => ({ sessions: [] }), refreshTranscript)
    noteSessionEvent('rt-screen')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    expect(refreshTranscript).toHaveBeenCalledWith('rt-screen', 's-screen')
    expect(card('rt-screen')).toBeUndefined()
  })

  it('leaves a background session to read its history when opened', async () => {
    publishSessionState('rt-bg', state({ awaitingResponse: true, busy: true, storedSessionId: 's-bg', turnLive: true }))
    const refreshTranscript = vi.fn()
    backend(async () => ({ sessions: [] }), refreshTranscript)
    noteSessionEvent('rt-bg')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    expect(refreshTranscript).not.toHaveBeenCalled()
    expect($workingSessionIds.get()).not.toContain('s-bg')
  })

  it('asks the backend that owns the turn, not the window gateway', async () => {
    publishSessionState('rt-remote', partial('partial', { storedSessionId: 's-remote' }))
    recordSessionEventScope({ connectionId: 'homelab', profile: 'work', session_id: 'rt-remote' })
    vi.mocked(requestGatewayForAgent).mockResolvedValue(listing('rt-remote', 'working') as never)
    const ambient = backend(async () => ({ sessions: [] }))
    noteSessionEvent('rt-remote')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    expect(ambient).not.toHaveBeenCalled()
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'homelab',
      'work',
      'session.active_list',
      {},
      LIVE_TURN_PROBE_TIMEOUT_MS,
      undefined
    )
    expect($workingSessionIds.get()).toContain('s-remote')
  })

  it('writes the settle through the wiring cache that holds the session', async () => {
    publishSessionState('rt-held', partial('the answer', { storedSessionId: 's-held' }))

    const updateHeldSession = vi.fn((runtimeId: string, updater: (s: ClientSessionState) => ClientSessionState) => {
      publishSessionState(runtimeId, updater($sessionStates.get()[runtimeId]!))

      return true
    })

    setSessionTileDelegate({ updateHeldSession } as unknown as SessionTileDelegate)
    backend(async () => ({ sessions: [] }))
    noteSessionEvent('rt-held')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    expect(updateHeldSession).toHaveBeenCalledWith('rt-held', expect.any(Function))
    expect($workingSessionIds.get()).not.toContain('s-held')
  })

  it('does not check a turn the user is still answering', async () => {
    publishSessionState('rt-ask', partial('need a choice', { needsInput: true, storedSessionId: 's-ask' }))
    const request = backend(async () => ({ sessions: [] }))
    noteSessionEvent('rt-ask')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    expect(request).not.toHaveBeenCalled()
    expect($workingSessionIds.get()).toContain('s-ask')
  })

  it('does not check a turn that settled before the silence window', async () => {
    const working = partial('done soon', { storedSessionId: 's-soon' })
    const request = backend(async () => ({ sessions: [] }))
    publishSessionState('rt-soon', working)
    noteSessionEvent('rt-soon')
    publishSessionState('rt-soon', {
      ...working,
      awaitingResponse: false,
      busy: false,
      messages: working.messages.map(message => ({ ...message, pending: false })),
      turnLive: false
    })

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    expect(request).not.toHaveBeenCalled()
    expect(card('rt-soon')).toBeUndefined()
  })
})

describe('liveTurnVerdict', () => {
  it('reads one runtime out of an active-list snapshot', () => {
    const snapshot = {
      sessions: [
        { id: 'a', status: 'working' },
        { id: 'b', status: 'starting' },
        { id: 'c', status: 'waiting' },
        { id: 'd', status: 'idle' },
        { id: 'e', status: 'from-a-newer-backend' }
      ]
    }

    expect(liveTurnVerdict(snapshot, 'a')).toBe('running')
    expect(liveTurnVerdict(snapshot, 'b')).toBe('running')
    expect(liveTurnVerdict(snapshot, 'c')).toBe('running')
    expect(liveTurnVerdict(snapshot, 'd')).toBe('ended')
    expect(liveTurnVerdict(snapshot, 'gone')).toBe('ended')
    expect(liveTurnVerdict(snapshot, 'e')).toBe('unknown')
    expect(liveTurnVerdict({}, 'a')).toBe('unknown')
    expect(liveTurnVerdict(null, 'a')).toBe('unknown')
  })
})
