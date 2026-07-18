import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import type { TodoItem } from '@/lib/todos'
import { $messages } from '@/store/session'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'

import { SessionTaskHistoryPopover, TaskHistoryPopover } from './task-history-popover'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => vi.stubGlobal('ResizeObserver', TestResizeObserver))

const list = (status: TodoItem['status'], content = 'Ship feature'): TodoItem[] => [{ content, id: content, status }]

const message = (id: string, todos: TodoItem[], timestamp?: number): ChatMessage => ({
  id,
  parts: [
    {
      args: { todos: todos.map(({ content, id: todoId, status }) => ({ content, id: todoId, status })) },
      toolCallId: `${id}-todo`,
      toolName: 'todo',
      type: 'tool-call'
    }
  ],
  role: 'assistant',
  timestamp
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  $messages.set([])
  clearSessionTodos('linger')
})

describe('TaskHistoryPopover', () => {
  it('does not render a button when neither transcript nor live state has a list', () => {
    render(<TaskHistoryPopover busy={false} liveTodos={null} messages={[]} sessionId="a" />)

    expect(screen.queryByRole('button', { name: /Tasks/ })).toBeNull()
  })

  it('opens, closes from the UI, and reopens without deleting transcript history', () => {
    render(
      <TaskHistoryPopover
        busy={false}
        liveTodos={null}
        messages={[message('turn-1', list('completed'))]}
        sessionId="a"
      />
    )

    const trigger = screen.getByRole('button', { name: 'Tasks 1/1' })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(trigger)
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByRole('dialog', { name: 'Task history' })).not.toBeNull()
    expect(screen.getByText('Ship feature')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Close task history' }))
    expect(screen.queryByRole('dialog', { name: 'Task history' })).toBeNull()

    fireEvent.click(trigger)
    expect(screen.getByText('Ship feature')).not.toBeNull()
  })

  it('scopes open presentation state to each runtime session', () => {
    const { rerender } = render(
      <TaskHistoryPopover
        busy={false}
        liveTodos={null}
        messages={[message('a-turn', list('completed', 'A task'))]}
        sessionId="a"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Tasks 1/1' }))
    expect(screen.getByRole('dialog', { name: 'Task history' })).not.toBeNull()

    rerender(
      <TaskHistoryPopover
        busy={false}
        liveTodos={null}
        messages={[message('b-turn', list('completed', 'B task'))]}
        sessionId="b"
      />
    )
    expect(screen.queryByRole('dialog', { name: 'Task history' })).toBeNull()
    expect(screen.queryByText('A task')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Tasks 1/1' }))
    expect(screen.getByText('B task')).not.toBeNull()

    rerender(
      <TaskHistoryPopover
        busy={false}
        liveTodos={null}
        messages={[message('a-turn', list('completed', 'A task'))]}
        sessionId="a"
      />
    )
    expect(screen.getByText('A task')).not.toBeNull()
    expect(screen.queryByText('B task')).toBeNull()
  })

  it('reconstructs history when a resumed transcript arrives', () => {
    const { rerender } = render(<TaskHistoryPopover busy={false} liveTodos={null} messages={[]} sessionId="resume" />)
    expect(screen.queryByRole('button', { name: /Tasks/ })).toBeNull()

    rerender(
      <TaskHistoryPopover
        busy={false}
        liveTodos={null}
        messages={[message('old-turn', list('in_progress', 'Interrupted task'))]}
        sessionId="resume"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Tasks 0/1' }))
    expect(screen.getByText('Unfinished from previous turn')).not.toBeNull()
    expect(screen.queryByText('Running')).toBeNull()
  })

  it('shows turn timestamps so older lists remain understandable', () => {
    const firstTimestamp = 1_752_840_000
    const secondTimestamp = 1_752_843_600
    const latestTimestamp = 1_752_847_200

    render(
      <TaskHistoryPopover
        busy={false}
        liveTodos={null}
        messages={[
          message('turn-1', list('completed', 'First list'), firstTimestamp),
          message('turn-2', list('completed', 'Second list'), secondTimestamp),
          message('turn-3', list('completed', 'Latest list task'), latestTimestamp)
        ]}
        sessionId="history"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Tasks 1/1' }))

    expect(screen.getByRole('heading', { name: 'Latest list' })).not.toBeNull()
    expect(screen.getByText(new Date(latestTimestamp * 1000).toLocaleString())).not.toBeNull()
    expect(screen.getByText(new Date(secondTimestamp * 1000).toLocaleString())).not.toBeNull()
    expect(screen.getByText(new Date(firstTimestamp * 1000).toLocaleString())).not.toBeNull()
  })

  it('keeps completed history after the four-second live linger clears', () => {
    vi.useFakeTimers()
    $messages.set([message('turn-1', list('completed'))])
    setSessionTodos('linger', list('completed'))
    render(<SessionTaskHistoryPopover busy={false} sessionId="linger" />)

    expect(screen.getByRole('button', { name: 'Tasks 1/1' })).not.toBeNull()

    act(() => vi.advanceTimersByTime(4_000))

    expect($todosBySession.get().linger).toBeUndefined()
    expect(screen.getByRole('button', { name: 'Tasks 1/1' })).not.toBeNull()
  })

  it('reveals long history progressively', () => {
    const messages = Array.from({ length: 12 }, (_, index) =>
      message(`turn-${index}`, list('completed', `List ${index}`), 1_752_840_000 + index * 60)
    )

    render(<TaskHistoryPopover busy={false} liveTodos={null} messages={messages} sessionId="long" />)
    fireEvent.click(screen.getByRole('button', { name: 'Tasks 1/1' }))

    expect(screen.getByText('List 11')).not.toBeNull()
    expect(screen.queryByText('List 0')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Load older lists' }))

    expect(screen.getByText('List 0')).not.toBeNull()
  })
})
