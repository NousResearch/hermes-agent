import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

const complete = () =>
  act(() => stream.handleEvent({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' }))

describe('useMessageStream turn-end todo cleanup', () => {
  beforeEach(() => {
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos(SID)
    vi.restoreAllMocks()
  })

  it('drops a still-active task list when the turn completes', () => {
    mountStream()
    setSessionTodos(SID, [todo('a', 'completed'), todo('b', 'in_progress')])

    complete()

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('keeps a finished list on completion so its linger shows the final checkmarks', () => {
    mountStream()
    setSessionTodos(SID, [todo('a', 'completed')])

    complete()

    // Not cleared immediately — the finished-list linger still owns it.
    expect($todosBySession.get()[SID]).toHaveLength(1)
  })

  it('drops a still-active task list when the turn errors out', () => {
    mountStream()
    setSessionTodos(SID, [todo('a', 'in_progress')])

    act(() => stream.handleEvent({ payload: { message: 'boom' }, session_id: SID, type: 'error' }))

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('applies a dedicated todo snapshot immediately', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { revision: 3, todos: [todo('live', 'in_progress')] },
        session_id: SID,
        type: 'todo.updated'
      })
    )

    expect($todosBySession.get()[SID]?.[0]?.id).toBe('live')
  })
})
