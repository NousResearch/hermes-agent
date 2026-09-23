import { describe, expect, it, vi } from 'vitest'

const { refreshSupportedSessionControlAfterTurn } = vi.hoisted(() => ({
  refreshSupportedSessionControlAfterTurn: vi.fn(async () => undefined)
}))

vi.mock('@/store/session-control', () => ({ refreshSupportedSessionControlAfterTurn }))

import type { GatewayEventName } from '@hermes/shared'

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
  it('paints an attributed relay user message before the assistant stream', () => {
    const ctx = context('message.user')
    ctx.payload = {
      text: 'Message from 🤖 zara:\nhello',
      delivery_id: 'd'.repeat(32),
      author: { id: 'bot:zara', name: 'zara', is_bot: true }
    }
    const state = { messages: [], streamId: null }
    ctx.deps.updateSessionState = vi.fn((_sid, updater) => updater(state as never) as never)

    expect(handleMessageStreamEvent(ctx)).toBe(true)
    const updater = vi.mocked(ctx.deps.updateSessionState).mock.calls[0]?.[1]
    const next = updater?.(state as never) as { messages: Array<{ id: string; role: string; parts: unknown[] }> }
    expect(next.messages).toMatchObject([
      { id: `relay-${'d'.repeat(32)}`, role: 'user', parts: [{ type: 'text', text: 'Message from 🤖 zara:\nhello' }] }
    ])
  })

  it('refreshes only after message.complete, through the store seam', () => {
    expect(handleMessageStreamEvent(context('message.delta'))).toBe(true)
    expect(refreshSupportedSessionControlAfterTurn).not.toHaveBeenCalled()

    expect(handleMessageStreamEvent(context('message.complete'))).toBe(true)
    expect(refreshSupportedSessionControlAfterTurn).toHaveBeenCalledTimes(1)
    expect(refreshSupportedSessionControlAfterTurn).toHaveBeenCalledWith('s1')
  })
})
