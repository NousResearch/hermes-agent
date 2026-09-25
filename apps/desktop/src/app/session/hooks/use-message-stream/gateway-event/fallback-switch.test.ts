import { afterEach, describe, expect, it, vi } from 'vitest'

import { handleStatusEvent } from './status'
import type { GatewayEventContext } from './types'

function statusContext(text: string, kind = 'lifecycle') {
  const updateSessionState = vi.fn((_id: string, updater: (state: { messages: unknown[] }) => unknown) =>
    updater({ messages: [] })
  )
  const payload = { kind, text } as GatewayEventContext['payload']
  const ctx: GatewayEventContext = {
    deps: {
      compactedTurnRef: { current: new Set<string>() },
      failAssistantMessage: vi.fn(),
      flushQueuedDeltas: vi.fn(),
      hydrateFromStoredSession: vi.fn(),
      queryClient: { invalidateQueries: vi.fn() },
      sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState
    } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: 'sess-1', type: 'status.update' },
    explicitSid: 'sess-1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_100,
    payload,
    scheduleConfigRefresh: vi.fn(),
    sessionId: 'sess-1'
  }

  return { ctx, updateSessionState }
}

function systemText(updateSessionState: ReturnType<typeof statusContext>['updateSessionState']): string {
  const updater = updateSessionState.mock.calls[0]?.[1]
  const next = updater?.({ messages: [] }) as { messages: Array<{ parts: Array<{ text?: string }>; role: string }> } | undefined

  return next?.messages.find(message => message.role === 'system')?.parts[0]?.text ?? ''
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('desktop fallback switch', () => {
  it('shows a model fallback switch in the transcript', () => {
    const notice = '⚠️ Model fallback: wan2.7-image-pro via alibaba-token-plan unavailable (bad request); using qwen3.8-max via alibaba-token-plan.'
    const { ctx, updateSessionState } = statusContext(notice)

    expect(handleStatusEvent(ctx)).toBe(true)
    expect(systemText(updateSessionState)).toContain('qwen3.8-max')
    expect(systemText(updateSessionState)).toMatch(/fallback/i)
  })

  it('does not turn unrelated lifecycle status into a transcript line', () => {
    const { ctx, updateSessionState } = statusContext('⚠️ Rate limited — retrying')

    expect(handleStatusEvent(ctx)).toBe(true)
    expect(updateSessionState).not.toHaveBeenCalled()
  })
})
