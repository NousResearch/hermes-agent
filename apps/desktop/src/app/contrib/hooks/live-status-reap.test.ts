import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from '@/store/session'
import {
  $attentionSessionIds,
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  publishSessionState
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

  it('fires the unread "your turn" marker for a vanished background session', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-b', session_key: 'stored-b', status: 'working' }]
    })

    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($unreadFinishedSessionIds.get()).toEqual(['stored-b'])
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
      messages: [
        { id: 'a1', role: 'assistant', parts: [openTool], pending: false } as never
      ]
    })

    // Keep the runtime referenced so the settled state stays in the store
    // instead of being evicted as no-longer-needed.
    $activeSessionId.set('runtime-tools')

    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-tools', session_key: 'stored-tools', status: 'working' }]
    })
    rehydrateLiveSessionStatuses({ sessions: [] })

    const state = $sessionStates.get()['runtime-tools']

    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)
    expect((state.messages[0].parts[0] as { result?: unknown }).result).toBeDefined()
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
