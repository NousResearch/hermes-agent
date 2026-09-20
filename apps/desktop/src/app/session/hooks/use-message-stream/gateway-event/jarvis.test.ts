import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $jarvisUi, resetJarvisSession } from '@/app/jarvis/store'

import { publishJarvisGatewayEvent } from './jarvis'
import type { GatewayEventContext } from './types'

function context(overrides: Partial<GatewayEventContext> = {}): GatewayEventContext {
  return {
    deps: {
      sessionStateByRuntimeIdRef: { current: new Map() },
    } as GatewayEventContext['deps'],
    event: { session_id: 'active-session', type: 'message.start' },
    explicitSid: 'active-session',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_100,
    payload: { task_id: 'task-1' } as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: 'active-session',
    ...overrides,
  }
}

describe('publishJarvisGatewayEvent', () => {
  beforeEach(() => {
    resetJarvisSession('active-session')
  })

  it('maps a confirmed active event from the active source into Jarvis state', () => {
    publishJarvisGatewayEvent(context())

    expect($jarvisUi.get()).toMatchObject({
      sessionId: 'active-session',
      task: { id: 'task-1', phase: 'running' },
      activity: [
        {
          at: 1_700_000_100,
          sessionId: 'active-session',
          taskId: 'task-1',
          type: 'task.running',
        },
      ],
    })
  })

  it('ignores active-session events from an inactive source', () => {
    const previous = $jarvisUi.get()

    publishJarvisGatewayEvent(context({ fromActiveSource: () => false }))

    expect($jarvisUi.get()).toBe(previous)
  })

  it('ignores background session events', () => {
    const previous = $jarvisUi.get()

    publishJarvisGatewayEvent(
      context({
        event: { session_id: 'background-session', type: 'message.start' },
        explicitSid: 'background-session',
        isActiveEvent: false,
        sessionId: 'background-session',
      }),
    )

    expect($jarvisUi.get()).toBe(previous)
  })

  it('marks a task verified only when the backend emits message.complete', () => {
    publishJarvisGatewayEvent(
      context({
        event: { session_id: 'active-session', type: 'tool.complete' },
        payload: { task_id: 'task-1', tool_id: 'tool-1' } as GatewayEventContext['payload'],
      }),
    )

    expect($jarvisUi.get().task).toEqual({ id: 'task-1', phase: 'idle' })

    publishJarvisGatewayEvent(
      context({
        event: { session_id: 'active-session', type: 'message.complete' },
        payload: { task_id: 'task-1' } as GatewayEventContext['payload'],
      }),
    )

    expect($jarvisUi.get().task).toEqual({ id: 'task-1', phase: 'verified' })
    expect($jarvisUi.get().activity.map(event => event.type)).toEqual(['tool.completed', 'task.verified'])
  })

  it('maps an error message.complete to a failed task with backend detail', () => {
    publishJarvisGatewayEvent(
      context({
        event: { session_id: 'active-session', type: 'message.complete' },
        payload: {
          error: 'provider rejected the request',
          rendered: 'rendered fallback',
          status: 'error',
          task_id: 'task-1',
          text: 'text fallback',
        } as GatewayEventContext['payload'],
      }),
    )

    expect($jarvisUi.get().task).toEqual({ id: 'task-1', phase: 'failed' })
    expect($jarvisUi.get().activity.at(-1)).toMatchObject({
      detail: 'provider rejected the request',
      type: 'task.failed',
    })
  })
})
