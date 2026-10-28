import { describe, expect, it, vi } from 'vitest'

vi.mock('@/store/session-control', () => ({ refreshSupportedSessionControlAfterTurn: vi.fn(async () => undefined) }))

import type { GatewayEventName } from '@hermes/shared'

import { liveTurnSettledDuringResume } from '@/app/session/hooks/use-session-actions/resume-reconcile'

import { handleMessageStreamEvent } from './message-stream'
import type { GatewayEventContext } from './types'

function complete(sessionId: string): GatewayEventContext {
  const type: GatewayEventName = 'message.complete'

  return {
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef: { current: sessionId },
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
    explicitSid: sessionId,
    fromActiveSource: () => true,
    isActiveEvent: false,
    occurredAt: Date.now() / 1000,
    payload: { text: 'the whole answer', status: 'complete' },
    scheduleConfigRefresh: vi.fn(),
    sessionId
  }
}

// A reconnect re-resumes the open chat; the turn's message.complete can land on
// the new socket before the resume snapshot (cut while the turn still ran)
// does. Reconciling that older snapshot over the settled reply rewound it to
// the snapshot's partial text, often the first chunk.
describe('a reply settled while a resume is in flight', () => {
  it('is newer than that resume, which keeps the live transcript', () => {
    const resumeStartedAtS = Date.now() / 1000 - 1

    handleMessageStreamEvent(complete('runtime-a'))

    expect(liveTurnSettledDuringResume({ session_id: 'runtime-a' }, resumeStartedAtS, true)).toBe(true)
  })

  it('does not outrank a resume that started after it, another session, or a cold open', () => {
    handleMessageStreamEvent(complete('runtime-b'))
    const laterResumeS = Date.now() / 1000 + 1

    expect(liveTurnSettledDuringResume({ session_id: 'runtime-b' }, laterResumeS, true)).toBe(false)
    expect(liveTurnSettledDuringResume({ session_id: 'runtime-other' }, 0, true)).toBe(false)
    expect(liveTurnSettledDuringResume({ session_id: 'runtime-b' }, 0, false)).toBe(false)
  })
})
