import { type GatewayEvent } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $subagentsBySession, clearSessionSubagents } from '@/store/subagents'

import { handleToolEvent } from './tools'
import type { GatewayEventContext } from './types'

// #75505 root cause 2: a Stop interrupts the parent TURN, not the children.
// A delegation finishing in the background fires `subagent.complete` after the
// interrupt; the SUBAGENT_EVENT_TYPES guard dropped it and the row spun
// 'running' forever. Terminal completions must land; live progress must not.
const sid = 'stopped-session'

function subagentEvent(type: string, payload: Record<string, unknown>): GatewayEvent {
  return { type, session_id: sid, seq: 1, payload } as GatewayEvent
}

function deliver(event: GatewayEvent, interrupted: boolean) {
  const nativeSubagentSessionsRef = { current: new Set<string>([sid]) }

  return handleToolEvent({
    event,
    payload: event.payload,
    sessionId: sid,
    occurredAt: 1,
    isActiveEvent: false,
    deps: {
      flushQueuedDeltas: vi.fn(),
      nativeSubagentSessionsRef,
      sessionInterrupted: () => interrupted,
      updateSessionState: vi.fn(),
      upsertToolCall: vi.fn()
    }
  } as unknown as GatewayEventContext)
}

describe('subagent events across an interrupted session', () => {
  beforeEach(() => {
    deliver(
      subagentEvent('subagent.start', { status: 'running', subagent_id: 'sa-1', goal: 'audit files', task_index: 0 }),
      false
    )
  })

  afterEach(() => {
    clearSessionSubagents(sid)
    $subagentsBySession.set({})
  })

  it('a terminal subagent.complete still lands after Stop instead of leaving the row running forever', () => {
    deliver(
      subagentEvent('subagent.complete', { status: 'completed', subagent_id: 'sa-1', summary: 'done', task_index: 0 }),
      true
    )

    const row = $subagentsBySession.get()[sid]?.[0]
    expect(row?.status).toBe('completed')
    expect(row?.summary).toBe('done')
  })

  it('live progress and non-terminal completions stay suppressed after Stop', () => {
    deliver(
      subagentEvent('subagent.progress', {
        status: 'running',
        subagent_id: 'sa-1',
        text: 'still going',
        task_index: 0
      }),
      true
    )
    deliver(subagentEvent('subagent.complete', { status: 'running', subagent_id: 'sa-1', task_index: 0 }), true)

    const row = $subagentsBySession.get()[sid]?.[0]
    expect(row?.status).toBe('running')
    expect(row?.stream).toHaveLength(0)
  })

  it('a completion carrying no status stays behind the interrupted guard (fail closed, no zombie repaint)', () => {
    deliver(subagentEvent('subagent.complete', { subagent_id: 'sa-1', task_index: 0 }), true)

    // A payload without a recognized terminal status is not trusted past the
    // interrupt; the pane-gated subagent.list reconcile remains its fallback.
    expect($subagentsBySession.get()[sid]?.[0]?.status).toBe('running')
  })
})
