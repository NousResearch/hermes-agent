import { describe, expect, it } from 'vitest'
import { initialJarvisUiState, reduceJarvisEvent } from './projector'

describe('reduceJarvisEvent', () => {
  it('stops speaking without cancelling the active task', () => {
    const speaking = reduceJarvisEvent(initialJarvisUiState(), {
      type: 'voice.speaking', sessionId: 's1', taskId: 't1', at: 1,
    })
    const stopped = reduceJarvisEvent(speaking, {
      type: 'voice.stopped', sessionId: 's1', taskId: 't1', at: 2,
    })
    expect(stopped.voice).toBe('idle')
    expect(stopped.task.phase).toBe('running')
  })

  it('ignores stale events from another task', () => {
    const current = {
      ...initialJarvisUiState(),
      sessionId: 's1',
      task: { id: 't2', phase: 'running' as const },
    }
    const next = reduceJarvisEvent(current, {
      type: 'tool.completed', sessionId: 's1', taskId: 't1', at: 3, label: 'Old tool',
    })
    expect(next).toBe(current)
  })

  it('tracks and clears the active tool from tool events', () => {
    const started = reduceJarvisEvent(initialJarvisUiState(), {
      type: 'tool.started',
      sessionId: 's1',
      taskId: 't1',
      toolCallId: 'call-1',
      at: 3,
      label: 'Read file',
    })
    const completed = reduceJarvisEvent(started, {
      type: 'tool.completed',
      sessionId: 's1',
      taskId: 't1',
      toolCallId: 'call-1',
      at: 4,
      label: 'Read file',
    })

    expect(started.activeTool).toEqual({ id: 'call-1', label: 'Read file' })
    expect(completed.activeTool).toBeNull()
  })

  it('keeps only the latest 50 activity events', () => {
    const next = Array.from({ length: 51 }, (_, index) => ({
      type: 'task.running',
      sessionId: 's1',
      taskId: 't1',
      at: index,
    })).reduce(reduceJarvisEvent, initialJarvisUiState())

    expect(next.activity).toHaveLength(50)
    expect(next.activity[0].at).toBe(1)
  })
})
