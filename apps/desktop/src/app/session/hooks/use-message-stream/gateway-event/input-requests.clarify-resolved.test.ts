import { afterEach, describe, expect, it, vi } from 'vitest'

import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

// When one attached client answers a clarify, the gateway withdraws the request
// from the others with request.cancel reason 'resolved' (#136709). That client
// must drop the card but leave the pending tool part alone — the authoritative
// result arrives as tool.complete, and stamping it cancelled/unanswered here
// would stick if that upsert treats the part as already settled.
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))

interface SessionState {
  messages: {
    id: string
    pending: boolean
    parts: { type: string; toolCallId: string; toolName: string; args: Record<string, unknown> }[]
  }[]
  busy: boolean
  needsInput: boolean
  streamId: string | null
}

function parkedState(): SessionState {
  return {
    busy: true,
    messages: [
      {
        id: 'msg-1',
        pending: true,
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'srv-1',
            toolName: 'clarify',
            args: { questions: [{ question: 'Deploy to prod?' }] }
          }
        ]
      }
    ],
    needsInput: true,
    streamId: 'stream-1'
  }
}

function cancelContext(reason: string, updateSessionState: ReturnType<typeof vi.fn>): GatewayEventContext {
  const payload = { id: 'srv-1', method: 'clarify', reason }

  return {
    deps: { updateSessionState } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: 's1', type: 'request.cancel' },
    explicitSid: 's1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_000,
    payload: payload as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1'
  }
}

describe('clarify request.cancel', () => {
  afterEach(() => {
    clearClarifyRequest(undefined, 's1')
  })

  it("resolved: drops the card but leaves the pending part for tool.complete instead of stamping it cancelled", () => {
    setClarifyRequest({
      questions: [{ choices: null, multiSelect: false, qid: 'q1', question: 'Deploy to prod?' }],
      requestId: 'srv-1',
      sessionId: 's1'
    })

    const updateSessionState = vi.fn((_: string, updater: (s: SessionState) => SessionState) =>
      updater(parkedState())
    )

    expect(handleInputRequestEvent(cancelContext('resolved', updateSessionState))).toBe(true)
    expect($clarifyRequests.get()['s1']).toBeUndefined()

    const next = updateSessionState.mock.results[0]?.value as SessionState

    expect(next.needsInput).toBe(false)
    expect(next.messages).toHaveLength(1)
    const part = next.messages[0].parts[0] as Record<string, unknown>

    expect(part.completedAt).toBeUndefined()
    expect(part.result).toBeUndefined()
  })

  it('timeout: still settles the part as cancelled/unanswered', () => {
    setClarifyRequest({
      questions: [{ choices: null, multiSelect: false, qid: 'q1', question: 'Deploy to prod?' }],
      requestId: 'srv-1',
      sessionId: 's1'
    })

    const updateSessionState = vi.fn((_: string, updater: (s: SessionState) => SessionState) =>
      updater(parkedState())
    )

    expect(handleInputRequestEvent(cancelContext('timeout', updateSessionState))).toBe(true)

    const next = updateSessionState.mock.results[0]?.value as SessionState
    const part = next.messages[0].parts[0] as { completedAt?: number; result?: { outcome: string } }

    expect(part.completedAt).toBe(1_700_000_000)
    expect(part.result?.outcome).toBe('cancelled')
  })
})
