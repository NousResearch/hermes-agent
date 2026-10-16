import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { buildToolView } from '@/components/assistant-ui/tool/fallback-model'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from '@/store/session'
import {
  $attentionSessionIds,
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  publishSessionState,
  reconcileBusyStatesOnReconnect
} from '@/store/session-states'

import { rehydrateLiveSessionStatuses } from './use-background-sync'

/**
 * `session.active_list` is the authoritative snapshot of what is RUNNING in the
 * polled gateway process. A session that finished while Desktop was looking
 * elsewhere — or whose runtime id was recycled by a backend respawn — simply
 * stops appearing in the response. Absence is therefore a completion signal,
 * not "no news": if nothing reaps it, the row spins forever and the
 * busy→idle edge that paints the green "your turn" dot never fires.
 */
describe('rehydrateLiveSessionStatuses — reaping vanished runtimes', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $selectedStoredSessionId.set(null)
    $unreadFinishedSessionIds.set([])
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $activeSessionId.set(null)
  })

  it('clears a working session that disappears from the live snapshot', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-a', session_key: 'stored-a', status: 'working' }]
    })

    expect($workingSessionIds.get()).toEqual(['stored-a'])

    // The turn finished and the gateway reaped the session between polls.
    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($workingSessionIds.get()).toEqual([])
  })

  it('settles a retained runtime when the backend reports its turn idle', () => {
    $activeSessionId.set('runtime-a')
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-a', session_key: 'stored-a', status: 'working' }]
    })

    const openTool = {
      type: 'tool-call',
      toolCallId: 'call-idle',
      toolName: 'read_file',
      args: {},
      argsText: '{}'
    } as never

    publishSessionState('runtime-a', {
      ...$sessionStates.get()['runtime-a'],
      awaitingResponse: true,
      sawAssistantPayload: true,
      turnLive: true,
      streamId: 'reply-a',
      messages: [{ id: 'reply-a', role: 'assistant', parts: [openTool], pending: false } as never]
    })
    expect($workingSessionIds.get()).toEqual(['stored-a'])

    // The session still EXISTS; only the parent turn ended. No message.complete
    // arrived on this Desktop socket, but the owner-routed active_list did.
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-a', session_key: 'stored-a', status: 'idle' }]
    })

    expect($workingSessionIds.get()).toEqual([])
    expect($sessionStates.get()['runtime-a']).toMatchObject({
      busy: false,
      awaitingResponse: false,
      turnLive: false,
      streamId: null
    })

    const part = $sessionStates.get()['runtime-a'].messages[0].parts[0]
    expect(part.type).toBe('tool-call')

    if (part.type !== 'tool-call') {
      throw new Error('Expected a tool call')
    }

    expect(part.completedAt).toBeDefined()
    expect(part.result).toBeUndefined() // idle is not proof the tool succeeded
  })

  it('ignores a stale pre-turn idle snapshot after a newer turn observed its first payload', () => {
    // Poll issued while the backend was idle: stateAtRequest is the pre-turn
    // snapshot. It races the submit below, so its `idle` answer is stale the
    // moment it lands.
    publishSessionState('runtime-race2', { ...createClientSessionState('stored-race2') })
    const stateAtRequest = $sessionStates.get()

    // The turn starts and its first payload arrives while the poll is in
    // flight — sawAssistantPayload is already true, so localSubmitPending is
    // false and ONLY the request-time reference guard protects it.
    publishSessionState('runtime-race2', {
      ...$sessionStates.get()['runtime-race2'],
      busy: true,
      awaitingResponse: true,
      sawAssistantPayload: true,
      turnLive: true,
      streamId: 'reply-b'
    })

    // The old idle response finally arrives.
    rehydrateLiveSessionStatuses(
      { sessions: [{ id: 'runtime-race2', session_key: 'stored-race2', status: 'idle' }] },
      Date.now(),
      'default',
      stateAtRequest
    )

    expect($workingSessionIds.get()).toContain('stored-race2')
    expect($sessionStates.get()['runtime-race2']).toMatchObject({
      busy: true,
      awaitingResponse: true,
      sawAssistantPayload: true,
      turnLive: true,
      streamId: 'reply-b'
    })
    expect($unreadFinishedSessionIds.get()).not.toContain('stored-race2')

    // Positive control: a CURRENT idle snapshot (request issued after the same
    // live state) is not stale and must settle the turn.
    rehydrateLiveSessionStatuses(
      { sessions: [{ id: 'runtime-race2', session_key: 'stored-race2', status: 'idle' }] },
      Date.now(),
      'default',
      $sessionStates.get()
    )

    expect($workingSessionIds.get()).not.toContain('stored-race2')
    expect($sessionStates.get()['runtime-race2']).toMatchObject({
      busy: false,
      awaitingResponse: false,
      turnLive: false,
      streamId: null
    })
  })

  it('does not settle a just-submitted turn before its first payload', () => {
    publishSessionState('runtime-new', {
      ...createClientSessionState('stored-new'),
      busy: true,
      awaitingResponse: true,
      sawAssistantPayload: false,
      turnLive: true
    })

    // A poll issued before the backend accepted the turn may report idle.
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-new', session_key: 'stored-new', status: 'idle' }]
    })

    expect($workingSessionIds.get()).toEqual(['stored-new'])
    expect($sessionStates.get()['runtime-new']).toMatchObject({
      busy: true,
      awaitingResponse: true,
      turnLive: true
    })
  })

  it('fires the unread "your turn" marker for a vanished background session', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-b', session_key: 'stored-b', status: 'working' }]
    })

    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($unreadFinishedSessionIds.get()).toEqual(['stored-b'])
  })

  // A turn that started just before the socket dropped was never polled, so
  // "seen live last poll" cannot gate its confirmation: the reconcile parked it,
  // and the first fresh snapshot that does not report it working is the
  // terminal fact that lights the dot.
  it('confirms a parked reconnect completion the poll never saw live', () => {
    publishSessionState('runtime-p', {
      ...createClientSessionState('stored-p'),
      busy: true,
      storedSessionId: 'stored-p'
    })
    reconcileBusyStatesOnReconnect()
    expect($unreadFinishedSessionIds.get()).toEqual([])

    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($unreadFinishedSessionIds.get()).toEqual(['stored-p'])
  })

  it('clears a blocked session that disappears from the live snapshot', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-c', session_key: 'stored-c', status: 'waiting' }]
    })

    expect($attentionSessionIds.get()).toEqual(['stored-c'])

    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($attentionSessionIds.get()).toEqual([])
  })

  it('leaves runtimes this poll never seeded alone', () => {
    // A background PROFILE's sessions are served by a different gateway and
    // never appear in this profile's active_list. Reaping them would dark out
    // every other profile's running rows.
    rehydrateLiveSessionStatuses(
      { sessions: [{ id: 'runtime-other', session_key: 'stored-other', status: 'working' }] },
      Date.now(),
      'other'
    )

    rehydrateLiveSessionStatuses({ sessions: [] }, Date.now(), 'default')

    expect($workingSessionIds.get()).toEqual(['stored-other'])
  })

  it('seals open tool parts and clears awaitingResponse when a session vanishes', () => {
    const openTool = {
      type: 'tool-call',
      toolCallId: 'call-1',
      toolName: 'patch',
      args: {},
      argsText: '{}'
    } as never

    publishSessionState('runtime-tools', {
      ...createClientSessionState('stored-tools'),
      busy: true,
      awaitingResponse: true,
      messages: [{ id: 'a1', role: 'assistant', parts: [openTool], pending: false } as never]
    })

    // Keep the runtime referenced so the settled state stays in the store
    // instead of being evicted as no-longer-needed.
    $activeSessionId.set('runtime-tools')

    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-tools', session_key: 'stored-tools', status: 'working' }]
    })
    rehydrateLiveSessionStatuses({ sessions: [] })

    const state = $sessionStates.get()['runtime-tools']
    const part = state.messages[0].parts[0]

    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)
    expect(part.type).toBe('tool-call')

    if (part.type !== 'tool-call') {
      throw new Error('Missing tool call')
    }

    // Reaping ends liveness without inventing evidence of a successful result.
    expect(part.completedAt).toBeDefined()
    expect(part.result).toBeUndefined()
    expect(buildToolView(part, '').status).toBe('warning')
  })

  it('clears a session stuck awaiting a response without the busy flag', () => {
    publishSessionState('runtime-await', {
      ...createClientSessionState('stored-await'),
      awaitingResponse: true,
      busy: false
    })

    $activeSessionId.set('runtime-await')

    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-await', session_key: 'stored-await', status: 'working' }]
    })
    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($sessionStates.get()['runtime-await'].awaitingResponse).toBe(false)
  })
})
