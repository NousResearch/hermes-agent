import type { GatewayEvent } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { $toolProgress, $toolProgressFor, clearSessionToolProgress, formatToolProgress } from '@/store/tool-progress'

import { handleToolProgressEvent } from './tools'
import type { GatewayEventContext } from './types'

function deliver(event: GatewayEvent, sessionId = 's1') {
  return handleToolProgressEvent({
    event,
    payload: event.payload,
    sessionId,
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

const progress = (sessionId: string, percent: number, message: string) =>
  ({
    type: 'tool.progress',
    session_id: sessionId,
    payload: { tool_id: 'call_0', name: 'mcp__meshy__refine', progress: percent, total: 100, percent, message }
  }) as unknown as GatewayEvent

afterEach(() => $toolProgress.set({}))

it('shows tool.progress on its running row and drops it when the call completes', () => {
  expect(deliver(progress('s1', 42, 'refining'))).toBe(true)
  // MCP rows render each batched call as `<call id>:<index>`.
  expect(formatToolProgress($toolProgressFor('s1', 'call_0:0').get())).toBe('42% · refining')

  deliver({ type: 'tool.complete', session_id: 's1', payload: { tool_id: 'call_0', name: 'mcp__meshy__refine' } } as GatewayEvent)
  expect($toolProgressFor('s1', 'call_0:0').get()).toBeUndefined()
})

it('keeps two chats reusing one call id apart, and a turn end sweeps only its own chat', () => {
  deliver(progress('s1', 42, 'refining'), 's1')
  deliver(progress('s2', 7, 'previewing'), 's2')

  expect(formatToolProgress($toolProgressFor('s1', 'call_0').get())).toBe('42% · refining')
  expect(formatToolProgress($toolProgressFor('s2', 'call_0').get())).toBe('7% · previewing')

  deliver({ type: 'tool.complete', session_id: 's2', payload: { tool_id: 'call_0' } } as GatewayEvent, 's2')
  expect($toolProgressFor('s2', 'call_0').get()).toBeUndefined()
  expect(formatToolProgress($toolProgressFor('s1', 'call_0').get())).toBe('42% · refining')

  clearSessionToolProgress('s1')
  expect($toolProgressFor('s1', 'call_0').get()).toBeUndefined()
})
