import { describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

// `_emit_approval_request` (tui_gateway/server.py) stamps `stored_session_id`
// with the durable session key alongside the event's own `session_id`, which
// is the ephemeral `ui_session` handle the gateway pairs with that key at
// accept time (tui_gateway/server.py `_wire_session_agent`). Neither a tile
// nor a session row nor an owner hint is ever keyed by that handle, so on a
// multi-profile install `approval.respond`'s owner-resolution ladder can only
// find the right backend once `storedSessionIdForRuntimeId` can translate the
// handle — which it does by reading `$sessionStates[sessionId].storedSessionId`,
// stamped via `updateSessionState`'s third argument (#105469).
function context(overrides: Partial<GatewayEventContext> = {}): GatewayEventContext {
  return {
    deps: {
      activeSessionIdRef: { current: null },
      sessionInterrupted: () => false,
      updateSessionState: vi.fn((_sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) =>
        updater({ needsInput: false } as ClientSessionState)
      ),
      upsertToolCall: vi.fn()
    },
    event: { session_id: 'ui-handle-1', type: 'approval.request' },
    explicitSid: 'ui-handle-1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_100,
    payload: { command: 'rm -rf /tmp/scratch' },
    scheduleConfigRefresh: vi.fn(),
    sessionId: 'ui-handle-1',
    ...overrides
  } as unknown as GatewayEventContext
}

describe('handleInputRequestEvent approval.request', () => {
  it('stamps the durable session key from the payload onto the ephemeral handle', () => {
    const ctx = context({ payload: { command: 'rm -rf /tmp/scratch', stored_session_id: 'stored-durable-id' } })

    expect(handleInputRequestEvent(ctx)).toBe(true)

    expect(ctx.deps.updateSessionState).toHaveBeenCalledWith(
      'ui-handle-1',
      expect.any(Function),
      'stored-durable-id'
    )
  })

  it('still flags needsInput when the backend omits stored_session_id (older gateway)', () => {
    const ctx = context()

    expect(handleInputRequestEvent(ctx)).toBe(true)

    expect(ctx.deps.updateSessionState).toHaveBeenCalledWith('ui-handle-1', expect.any(Function), undefined)

    const updater = (ctx.deps.updateSessionState as ReturnType<typeof vi.fn>).mock.calls[0][1] as (
      state: ClientSessionState
    ) => ClientSessionState

    expect(updater({ needsInput: false } as ClientSessionState)).toMatchObject({ needsInput: true })
  })
})
