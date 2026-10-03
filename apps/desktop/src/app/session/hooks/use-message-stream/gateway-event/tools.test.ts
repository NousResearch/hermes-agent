import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { handleToolEvent } from './tools'
import type { GatewayEventContext } from './types'

const context = (type: string, payload: Record<string, unknown>): GatewayEventContext =>
  ({
    deps: {
      flushQueuedDeltas: vi.fn(),
      nativeSubagentSessionsRef: { current: new Set<string>() },
      sessionInterrupted: vi.fn(() => false),
      updateSessionState: vi.fn(),
      upsertToolCall: vi.fn()
    },
    event: { type },
    isActiveEvent: false,
    occurredAt: 1,
    payload,
    sessionId: 'parent'
  }) as unknown as GatewayEventContext

describe('handleToolEvent delegate ownership', () => {
  beforeEach(() => $subagentsBySession.set({}))

  it('promotes one unique fallback on a creation-capable native event', () => {
    upsertSubagent('parent', {
      goal: 'owned goal',
      status: 'running',
      subagent_id: 'delegate-tool:call-owned:0',
      task_index: 0
    })

    handleToolEvent(
      context('subagent.start', {
        goal: 'owned goal',
        status: 'running',
        subagent_id: 'native-owned',
        task_index: 0
      })
    )

    expect($subagentsBySession.get().parent).toMatchObject([
      { delegateCallId: 'call-owned', delegateRowIndex: 0, id: 'native-owned' }
    ])
  })

  it('preserves the matched fallback row when child index 1 starts first', () => {
    upsertSubagent('parent', {
      goal: 'first goal',
      status: 'running',
      subagent_id: 'delegate-tool:call-batch:0',
      task_index: 0
    })
    upsertSubagent('parent', {
      goal: 'second goal',
      status: 'running',
      subagent_id: 'delegate-tool:call-batch:1',
      task_index: 1
    })

    handleToolEvent(
      context('subagent.start', {
        goal: 'second goal',
        status: 'running',
        subagent_id: 'native-second',
        task_index: 1
      })
    )

    expect($subagentsBySession.get().parent).toMatchObject([
      { delegateCallId: 'call-batch', delegateRowIndex: 1, id: 'native-second' }
    ])
  })

  it.each(['subagent.progress', 'subagent.complete'])(
    'does not promote or create a child from a %s event',
    type => {
      upsertSubagent('parent', {
        goal: 'legacy goal',
        status: 'running',
        subagent_id: 'delegate-tool:call-legacy:0',
        task_index: 0
      })

      handleToolEvent(
        context(type, {
          goal: 'legacy goal',
          status: type === 'subagent.complete' ? 'completed' : 'running',
          subagent_id: 'native-late',
          task_index: 0
        })
      )

      expect($subagentsBySession.get().parent).toMatchObject([
        { delegateCallId: undefined, id: 'delegate-tool:call-legacy:0' }
      ])
    }
  )
})
