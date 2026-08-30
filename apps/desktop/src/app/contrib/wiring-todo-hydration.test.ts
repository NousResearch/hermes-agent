import { afterEach, describe, expect, it } from 'vitest'

import {
  $todosBySession,
  applyTodoContinuationSnapshot,
  clearAllSessionTodos,
  clearAllTodoContinuations
} from '@/store/todos'

import { hydrateSessionTodos } from './wiring-todo-hydration'

const runtimeSessionId = 'runtime-target'

const unfinishedHistory = [
  {
    parts: [
      {
        args: { todos: [{ content: 'finish the task', id: 'task', status: 'in_progress' }] },
        toolCallId: 'todo-1',
        toolName: 'todo',
        type: 'tool-call'
      }
    ]
  }
]

afterEach(() => {
  clearAllSessionTodos()
  clearAllTodoContinuations()
})

describe('wiring todo hydration caller', () => {
  it.each(['active', 'paused'] as const)(
    'preserves unfinished history for the exact runtime when continuation is %s',
    state => {
      applyTodoContinuationSnapshot(runtimeSessionId, { revision: 1, state })

      hydrateSessionTodos(runtimeSessionId, unfinishedHistory)

      expect($todosBySession.get()[runtimeSessionId]).toEqual([
        { content: 'finish the task', id: 'task', status: 'in_progress' }
      ])
    }
  )

  it('does not use another runtime continuation to restore unfinished history', () => {
    applyTodoContinuationSnapshot('runtime-other', { revision: 1, state: 'active' })

    hydrateSessionTodos(runtimeSessionId, unfinishedHistory)

    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
  })

  it.each(['absent', 'none'] as const)('clears stale unfinished history when continuation is %s', state => {
    if (state === 'none') {
      applyTodoContinuationSnapshot(runtimeSessionId, { revision: 1, state: 'none' })
    }

    hydrateSessionTodos(runtimeSessionId, unfinishedHistory)

    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
  })
})
