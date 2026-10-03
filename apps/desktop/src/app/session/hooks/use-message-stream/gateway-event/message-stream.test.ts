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

describe('message.user_echo (#55564)', () => {
  type EchoState = { messages: { id: string; role: string; rowId?: number }[]; streamId: null }

  function echo(payload: Record<string, unknown>, state: EchoState, seq = 7) {
    const ctx = context('message.user_echo')
    ctx.event = { type: 'message.user_echo', seq } as GatewayEventContext['event']
    ctx.payload = payload as GatewayEventContext['payload']
    expect(handleMessageStreamEvent(ctx)).toBe(true)

    const update = vi.mocked(ctx.deps.updateSessionState).mock.calls[0]?.[1] as unknown as
      | ((s: EchoState) => EchoState & { busy?: boolean })
      | undefined

    return update ? update(state) : null
  }

  const empty = (): EchoState => ({ messages: [], streamId: null })

  it('seeds a user bubble for a prompt submitted by another client', () => {
    const next = echo({ text: 'hi from elsewhere', row_id: 42 }, empty())

    expect(next?.messages).toHaveLength(1)
    expect(next?.messages[0]).toMatchObject({ role: 'user', rowId: 42 })
    expect(next?.busy).toBe(true)
  })

  it('does not double a bubble on replay or when the row is already hydrated', () => {
    const once = echo({ text: 'hi', row_id: 42 }, empty())!
    expect(echo({ text: 'hi', row_id: 42 }, once)?.messages).toHaveLength(1)

    const hydrated: EchoState = { messages: [{ id: '1700-0-user', role: 'user', rowId: 42 }], streamId: null }
    expect(echo({ text: 'hi', row_id: 42 }, hydrated)?.messages).toHaveLength(1)
  })

  it('ignores hidden and empty echoes', () => {
    expect(echo({ text: 'scaffolding', display_kind: 'hidden' }, empty())).toBeNull()
    expect(echo({ text: '   ' }, empty())).toBeNull()
  })
})
