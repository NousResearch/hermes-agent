import { describe, expect, it, vi } from 'vitest'

import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

function context(type: string, payload: Record<string, unknown> = {}): GatewayEventContext {
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
    payload,
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1'
  }
}

describe('handleInputRequestEvent interrupted-session race', () => {
  it('parses and stores clarify.request even when session is interrupted (auto-continue deadlock fix)', () => {
    clearClarifyRequest(undefined, 's1')

    const ctx = context('clarify.request', {
      request_id: 'clarify_race_r1',
      question: 'Which option?',
      choices: ['continue', 'skip', 'stop']
    })

    // Simulate interrupted=true from a prior Stop
    ctx.deps.sessionInterrupted = vi.fn(() => true)
    const handled = handleInputRequestEvent(ctx)
    expect(handled).toBe(true)
    const stored = $clarifyRequests.get()['s1']
    expect(stored).toBeDefined()
    expect(stored?.requestId).toBe('clarify_race_r1')
    expect(stored?.question).toBe('Which option?')
    expect(stored?.choices).toEqual(['continue', 'skip', 'stop'])
  })

  it('handles batch clarify.request with interrupted flag set', () => {
    clearClarifyRequest(undefined, 's2')

    const ctx = context('clarify.request', {
      request_id: 'batch_race_r2',
      questions: [
        { qid: 'q0', question: 'First question?', choices: ['yes', 'no'] },
        { qid: 'q1', question: 'Second question?', choices: [] }
      ]
    })

    ctx.sessionId = 's2'
    ctx.explicitSid = 's2'
    ctx.deps.sessionInterrupted = vi.fn(() => true)
    const handled = handleInputRequestEvent(ctx)
    expect(handled).toBe(true)
    const stored = $clarifyRequests.get()['s2']
    expect(stored).toBeDefined()
    expect(stored?.requestId).toBe('batch_race_r2')
    expect(stored?.questions).toHaveLength(2)
    expect(stored?.questions?.[0].qid).toBe('q0')
    expect(stored?.questions?.[1].qid).toBe('q1')
  })
})
