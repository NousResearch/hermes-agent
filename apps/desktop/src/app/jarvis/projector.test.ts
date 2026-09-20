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

  it('clears a verified result when a new task starts planning or running', () => {
    const verified = reduceJarvisEvent({ ...initialJarvisUiState(), sessionId: 's1' }, {
      type: 'task.verified',
      sessionId: 's1',
      taskId: 't1',
      at: 1,
      detail: 'Notatka została utworzona.',
    })

    expect(verified.result).toBe('Notatka została utworzona.')

    const planning = reduceJarvisEvent(verified, {
      type: 'task.planning',
      sessionId: 's1',
      taskId: 't2',
      at: 2,
    })

    expect(planning.result).toBeUndefined()
    expect(planning.task).toEqual({ id: 't2', phase: 'planning' })

    const verifiedAgain = reduceJarvisEvent(planning, {
      type: 'task.verified',
      sessionId: 's1',
      taskId: 't2',
      at: 3,
      label: 'Kalendarz zaktualizowany.',
    })

    const running = reduceJarvisEvent(verifiedAgain, {
      type: 'task.running',
      sessionId: 's1',
      taskId: 't3',
      at: 4,
    })

    expect(running.result).toBeUndefined()
    expect(running.task).toEqual({ id: 't3', phase: 'running' })
  })

  it('does not treat a raw tool call id as the verified result headline', () => {
    const verified = reduceJarvisEvent({ ...initialJarvisUiState(), sessionId: 's1' }, {
      type: 'task.verified',
      sessionId: 's1',
      taskId: 't1',
      toolCallId: 'call_technical_123',
      at: 1,
    })

    expect(verified.result).toBeUndefined()
  })

  it('clears a verified result on terminal failure or cancellation', () => {
    const verified = reduceJarvisEvent({ ...initialJarvisUiState(), sessionId: 's1' }, {
      type: 'task.verified',
      sessionId: 's1',
      taskId: 't1',
      at: 1,
      detail: 'Gotowe.',
    })

    const failed = reduceJarvisEvent(verified, {
      type: 'task.failed',
      sessionId: 's1',
      taskId: 't1',
      at: 2,
      detail: 'Backend failed.',
    })

    expect(failed.result).toBeUndefined()

    const verifiedAgain = reduceJarvisEvent(failed, {
      type: 'task.verified',
      sessionId: 's1',
      taskId: 't1',
      at: 3,
      label: 'Gotowe ponownie.',
    })

    const cancelled = reduceJarvisEvent(verifiedAgain, {
      type: 'task.cancelled',
      sessionId: 's1',
      taskId: 't1',
      at: 4,
    })

    expect(cancelled.result).toBeUndefined()
  })

  it('keeps the current result when an irrelevant old session or old task event arrives', () => {
    const current = {
      ...initialJarvisUiState(),
      activity: [{ at: 10, detail: 'Aktualny wynik.', sessionId: 's1', taskId: 't2', type: 'task.verified' }],
      result: 'Aktualny wynik.',
      sessionId: 's1',
      task: { id: 't2', phase: 'verified' as const },
    }

    const oldSession = reduceJarvisEvent(current, {
      type: 'task.running',
      sessionId: 'old-session',
      taskId: 't2',
      at: 1,
    })

    const oldTask = reduceJarvisEvent(current, {
      type: 'task.running',
      sessionId: 's1',
      taskId: 'old-task',
      at: 2,
    })

    expect(oldSession).toBe(current)
    expect(oldTask).toBe(current)
    expect(oldSession.result).toBe('Aktualny wynik.')
    expect(oldTask.result).toBe('Aktualny wynik.')
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
