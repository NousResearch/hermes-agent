import { beforeEach, describe, expect, it, vi } from 'vitest'

const { refreshSupportedSessionControlAfterTurn } = vi.hoisted(() => ({
  refreshSupportedSessionControlAfterTurn: vi.fn(async () => undefined)
}))

vi.mock('@/store/session-control', () => ({ refreshSupportedSessionControlAfterTurn }))

import type { GatewayEventName } from '@hermes/shared'

import { CanonicalDesktopProtocol } from '@/api/canonical-protocol'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $lastReadAtBySessionId,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  setSelectedStoredSessionId
} from '@/store/session'
import { $sessionStates, publishSessionState } from '@/store/session-states'
import { $unreadFinishedMarkers, noteDesktopAdmission } from '@/store/session-unread'
import { makeSessionInfo } from '@/test/session-info'

import { handleMessageStreamEvent } from './message-stream'
import type { GatewayEventContext } from './types'

function context(type: GatewayEventName): GatewayEventContext {
  return {
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef: { current: 's1' },
      appendAssistantDelta: vi.fn(),
      appendReasoningDelta: vi.fn(),
      compactedTurnRef: { current: new Set() },
      completeAssistantMessage: vi.fn(),
      failAssistantMessage: vi.fn(),
      finalizeInterimAssistantMessage: vi.fn(),
      flushQueuedDeltas: vi.fn(),
      dropQueuedDeltas: vi.fn(),
      hydrateFromStoredSession: vi.fn(async () => undefined),
      lastCwdInfoSessionRef: { current: null },
      nativeSubagentSessionsRef: { current: new Set() },
      queryClient: {} as GatewayEventContext['deps']['queryClient'],
      refreshHermesConfig: vi.fn(async () => undefined),
      scheduleSessionsRefresh: vi.fn(),
      sessionInterrupted: vi.fn(() => false),
      sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState: vi.fn(),
      upsertToolCall: vi.fn()
    },
    event: { type },
    explicitSid: 's1',
    fromActiveSource: () => true,
    isActiveEvent: false,
    occurredAt: 1_700_000_100,
    payload: { text: 'completed' },
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1'
  }
}

describe('handleMessageStreamEvent session-control integration', () => {
  it('refreshes only after message.complete, through the store seam', () => {
    expect(handleMessageStreamEvent(context('message.delta'))).toBe(true)
    expect(refreshSupportedSessionControlAfterTurn).not.toHaveBeenCalled()

    expect(handleMessageStreamEvent(context('message.complete'))).toBe(true)
    expect(refreshSupportedSessionControlAfterTurn).toHaveBeenCalledTimes(1)
    expect(refreshSupportedSessionControlAfterTurn).toHaveBeenCalledWith('s1')
  })
})

describe('unread dot for a messaging turn Desktop sent', () => {
  beforeEach(() => {
    $sessionStates.set({})
    $sessions.set([])
    $messagingSessions.set([makeSessionInfo({ id: 'tg-1', source: 'telegram' })])
    $selectedStoredSessionId.set(null)
    $unreadFinishedSessionIds.set([])
    $unreadFinishedMarkers.set({})
    $lastReadAtBySessionId.set({})
    $activeGatewayProfile.set('default')
  })

  const complete = (admission?: string, sessionId = 'tg-1', profile?: string) => {
    const ctx = context('message.complete')
    ctx.sessionId = sessionId
    ctx.event = { type: 'message.complete', admission_id: admission, profile }
    handleMessageStreamEvent(ctx)
  }

  // A receipt as the canonical client sees it, whichever surface sent the prompt.
  const admit = (admission: string, status = 'queued') =>
    new CanonicalDesktopProtocol().result(
      'prompt.submit',
      { session_id: 'tg-1', submission_id: admission },
      {
        admission_id: admission,
        ref: { session_id: 'tg-1' },
        status
      }
    )

  it('lights on the stamped completion of a turn Desktop submitted, once', () => {
    noteDesktopAdmission('desktop-turn')
    complete('desktop-turn')
    expect($unreadFinishedSessionIds.get()).toContain('tg-1')

    $unreadFinishedSessionIds.set([])
    complete('desktop-turn')
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-1')
  })

  it('stays dark for a turn from the chat itself, and keeps a queued Desktop turn eligible', () => {
    noteDesktopAdmission('first')
    noteDesktopAdmission('queued-second')
    complete('chat-turn')
    complete('first')
    $unreadFinishedSessionIds.set([])

    complete('chat-turn-2')
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-1')
    complete('queued-second')
    expect($unreadFinishedSessionIds.get()).toContain('tg-1')
  })

  it('stays dark while the conversation is on screen', () => {
    noteDesktopAdmission('desktop-turn')
    $selectedStoredSessionId.set('tg-1')
    complete('desktop-turn')
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-1')
  })

  it('notes every canonical submit, so an edit, a regenerate or a Quick Entry send lights too', () => {
    admit('regenerated')
    complete('regenerated')
    expect($unreadFinishedSessionIds.get()).toContain('tg-1')
  })

  it('does not note a deduplicated submit that starts no turn', () => {
    admit('dedup', 'terminal')
    complete('dedup')
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-1')
  })

  it('lights a completion that arrived before its receipt, once the receipt lands', () => {
    complete('fast-turn')
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-1')
    admit('fast-turn')
    expect($unreadFinishedSessionIds.get()).toContain('tg-1')
  })

  it("judges a background profile's completion by its own profile's row", () => {
    // The active profile holds a Desktop session with the same id; it must not decide beta's turn.
    $activeGatewayProfile.set('alpha')
    $sessions.set([makeSessionInfo({ id: 'shared', source: 'desktop', profile: 'alpha' })])
    $messagingSessions.set([makeSessionInfo({ id: 'shared', source: 'telegram', profile: 'beta' })])
    noteDesktopAdmission('beta-turn')
    complete('beta-turn', 'shared', 'beta')
    expect($unreadFinishedSessionIds.get()).toContain('shared')
    expect($unreadFinishedMarkers.get()).toEqual({ beta: ['shared'] })
  })

  it('does not light a held completion the user read before its receipt landed', () => {
    complete('late-receipt')
    // Opened and left again while the receipt was in flight.
    setSelectedStoredSessionId('tg-1')
    setSelectedStoredSessionId(null)
    admit('late-receipt')
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-1')
    expect($unreadFinishedMarkers.get()).toEqual({})
  })

  it('does not light a compressed conversation whose listed segment the user read before the receipt landed', () => {
    $messagingSessions.set([makeSessionInfo({ id: 'tg-tip', source: 'telegram', _lineage_root_id: 'tg-root' })])
    complete('late-compressed', 'tg-root')
    setSelectedStoredSessionId('tg-tip')
    setSelectedStoredSessionId(null)
    admit('late-compressed')
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-tip')
    expect($unreadFinishedMarkers.get()).toEqual({})
  })

  it("does not mark another profile's same-id compressed conversation read", () => {
    // Opening alpha's local `tg-tip` must not stamp beta's root: its held completion still lights.
    $sessions.set([makeSessionInfo({ id: 'tg-tip', source: 'desktop', profile: 'alpha' })])
    $messagingSessions.set([
      makeSessionInfo({ id: 'tg-tip', source: 'telegram', _lineage_root_id: 'tg-root', profile: 'beta' })
    ])
    complete('beta-late', 'tg-root', 'beta')
    $activeGatewayProfile.set('alpha')
    setSelectedStoredSessionId('tg-tip')
    setSelectedStoredSessionId(null)
    expect($lastReadAtBySessionId.get()['tg-root']).toBeUndefined()
    admit('beta-late')
    expect($unreadFinishedSessionIds.get()).toContain('tg-tip')
    expect(Object.keys($unreadFinishedMarkers.get())).toEqual(['beta'])
  })

  it('marks the listed segment of a compressed conversation', () => {
    $messagingSessions.set([makeSessionInfo({ id: 'tg-tip', source: 'telegram', _lineage_root_id: 'tg-root' })])
    noteDesktopAdmission('compressed-turn')
    complete('compressed-turn', 'tg-root')
    expect($unreadFinishedSessionIds.get()).toContain('tg-tip')
  })

  it('never lights a messaging session from a busy-to-idle settle, whatever turn it follows', () => {
    noteDesktopAdmission('desktop-turn')
    publishSessionState('rt-tg', { ...createClientSessionState('tg-1'), busy: true })
    publishSessionState('rt-tg', { ...createClientSessionState('tg-1'), busy: false })
    expect($unreadFinishedSessionIds.get()).not.toContain('tg-1')
  })
})
