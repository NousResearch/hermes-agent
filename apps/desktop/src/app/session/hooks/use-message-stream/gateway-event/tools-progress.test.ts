import type { GatewayEvent } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { $toolProgress, $toolProgressFor, formatToolProgress } from '@/store/tool-progress'

import { handleToolProgressEvent } from './tools'
import type { GatewayEventContext } from './types'

function deliver(event: GatewayEvent) {
  return handleToolProgressEvent({
    event,
    payload: event.payload,
    sessionId: 's1',
    occurredAt: 1,
    isActiveEvent: true,
    deps: {
      flushQueuedDeltas: vi.fn(),
      updateSessionState: vi.fn(),
      sessionInterrupted: () => false,
      upsertToolCall: vi.fn(),
      nativeSubagentSessionsRef: { current: new Set() }
    }
  } as unknown as GatewayEventContext)
}

afterEach(() => $toolProgress.set({}))

it('shows tool.progress on its running row and drops it when the call completes', () => {
  const progress = {
    type: 'tool.progress',
    session_id: 's1',
    payload: { tool_id: 'c1', name: 'mcp__meshy__refine', progress: 42, total: 100, percent: 42, message: 'refining' }
  } as unknown as GatewayEvent

  expect(deliver(progress)).toBe(true)
  // MCP rows render each batched call as `<call id>:<index>`.
  expect(formatToolProgress($toolProgressFor('c1:0').get())).toBe('42% · refining')

  deliver({ type: 'tool.complete', session_id: 's1', payload: { tool_id: 'c1', name: 'mcp__meshy__refine' } } as GatewayEvent)
  expect($toolProgressFor('c1:0').get()).toBeUndefined()
})
