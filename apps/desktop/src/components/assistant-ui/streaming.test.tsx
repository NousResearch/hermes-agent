import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Thread } from './thread'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

const resizeObservers = new Set<TestResizeObserver>()

class TestResizeObserver {
  private target: Element | null = null

  constructor(private readonly callback: ResizeObserverCallback) {
    resizeObservers.add(this)
  }

  observe(target: Element) {
    this.target = target
  }

  unobserve() {}

  disconnect() {
    resizeObservers.delete(this)
  }

  trigger(height: number) {
    if (!this.target) {
      return
    }

    this.callback(
      [
        {
          contentRect: { height } as DOMRectReadOnly,
          target: this.target
        } as ResizeObserverEntry
      ],
      this as unknown as ResizeObserver
    )
  }
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return {
    cancel: () => {},
    finished: Promise.resolve()
  } as unknown as Animation
}

// jsdom returns 0 for offset*; the virtualizer reads those to size its
// viewport. Fall through to client* (which tests can override) or a sane
// default so virtualized items render.
function stubOffsetDimension(
  prop: 'offsetHeight' | 'offsetWidth',
  clientProp: 'clientHeight' | 'clientWidth',
  fallback: number
) {
  const previous = Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)

  Object.defineProperty(HTMLElement.prototype, prop, {
    configurable: true,
    get() {
      return previous?.get?.call(this) || (this as HTMLElement)[clientProp] || fallback
    }
  })
}

stubOffsetDimension('offsetWidth', 'clientWidth', 800)
stubOffsetDimension('offsetHeight', 'clientHeight', 600)

async function wait(ms: number) {
  await act(async () => {
    await new Promise(resolve => window.setTimeout(resolve, ms))
  })
}

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'Stream a response' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(text: string, running = true): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text }],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantErrorMessage(error: string): ThreadMessage {
  return {
    id: 'assistant-error-1',
    role: 'assistant',
    content: [],
    status: { type: 'incomplete', reason: 'error', error },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantReasoningMessage(text: string): ThreadMessage {
  return {
    id: 'assistant-reasoning-1',
    role: 'assistant',
    content: [{ type: 'reasoning', text }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantMultiReasoningMessage(texts: string[]): ThreadMessage {
  return {
    id: 'assistant-reasoning-multi-1',
    role: 'assistant',
    content: texts.map(text => ({ type: 'reasoning', text })),
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantTodoMessage(
  todos: Array<{ content: string; id: string; status: 'cancelled' | 'completed' | 'in_progress' | 'pending' }>,
  running = true
): ThreadMessage {
  const suffix = todos.map(todo => `${todo.id}:${todo.status}`).join('|') || 'empty'

  return {
    id: `assistant-todo-${running ? 'running' : 'done'}-${suffix}`,
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'todo-1',
        toolName: 'todo',
        args: { todos },
        argsText: JSON.stringify({ todos }),
        ...(running ? {} : { result: { todos } })
      }
    ],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantReasoningTodoMessage(
  todos: Array<{ content: string; id: string; status: 'cancelled' | 'completed' | 'in_progress' | 'pending' }>
): ThreadMessage {
  return {
    id: 'assistant-reasoning-todo-1',
    role: 'assistant',
    content: [
      { type: 'reasoning', text: 'Let me make a quick todo list.' },
      {
        type: 'tool-call',
        toolCallId: 'todo-1',
        toolName: 'todo',
        args: { todos },
        argsText: JSON.stringify({ todos }),
        result: { todos }
      },
      { type: 'text', text: 'Done — fake list created.' }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function StreamingHarness() {
  const [messages, setMessages] = useState<ThreadMessage[]>([userMessage()])
  const [isRunning, setIsRunning] = useState(true)

  useEffect(() => {
    const first = window.setTimeout(() => {
      setMessages([userMessage(), assistantMessage('first chunk')])
    }, 50)

    const second = window.setTimeout(() => {
      setMessages([userMessage(), assistantMessage('first chunk second chunk')])
    }, 500)

    const complete = window.setTimeout(() => {
      setMessages([userMessage(), assistantMessage('first chunk second chunk', false)])
      setIsRunning(false)
    }, 700)

    return () => {
      window.clearTimeout(first)
      window.clearTimeout(second)
      window.clearTimeout(complete)
    }
  }, [])

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread loading={isRunning && messages.at(-1)?.role !== 'assistant' ? 'response' : undefined} />
    </AssistantRuntimeProvider>
  )
}

function TodoHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: message.status?.type === 'running',
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function MessageHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function ReasoningHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantReasoningMessage(' The user is asking what this file is.')],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function GroupedReasoningHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantMultiReasoningMessage([' First thought.', ' Second thought.'])],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function IntroHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread intro={{ personality: 'default', seed: 1 }} />
    </AssistantRuntimeProvider>
  )
}

describe('assistant-ui streaming renderer', () => {
  beforeEach(() => {
    resizeObservers.clear()
  })

  it('renders assistant text incrementally before completion', async () => {
    const { container } = render(<StreamingHarness />)

    expect(screen.getByRole('status', { name: 'Hermes is loading a response' })).toBeTruthy()

    await wait(80)

    await waitFor(() => {
      expect(container.textContent).toContain('first chunk')
    })
    expect(container.textContent).not.toContain('second chunk')
    expect(screen.queryByRole('status', { name: 'Hermes is loading a response' })).toBeNull()

    await wait(500)

    await waitFor(() => {
      expect(container.textContent).toContain('first chunk second chunk')
    })

    await wait(250)

    await waitFor(() => {
      expect(container.textContent).toContain('first chunk second chunk')
    })
  })

  it('does not render composer clearance for intro-only threads', () => {
    const { container } = render(<IntroHarness />)

    expect(container.querySelector('[data-slot="aui_composer-clearance"]')).toBeNull()
  })

  it('renders assistant provider errors inline', () => {
    render(<MessageHarness message={assistantErrorMessage('OpenRouter rejected the request (403).')} />)

    expect(screen.getByRole('alert').textContent).toContain('OpenRouter rejected the request (403).')
  })

  it('does not pull the viewport back down after the user scrolls up during streaming', async () => {
    const { container } = render(<StreamingHarness />)

    const content = container.querySelector('[data-slot="aui_thread-content"]') as HTMLDivElement
    const viewport = content.parentElement as HTMLDivElement
    let scrollHeight = 1_000

    Object.defineProperty(viewport, 'clientHeight', { configurable: true, value: 200 })
    Object.defineProperty(viewport, 'scrollHeight', {
      configurable: true,
      get: () => scrollHeight
    })

    await wait(80)

    await act(async () => {
      viewport.scrollTop = 800
      fireEvent.scroll(viewport)
    })
    await wait(0)

    await act(async () => {
      fireEvent.wheel(viewport, { deltaY: -120 })
      viewport.scrollTop = 420
      fireEvent.scroll(viewport)
    })

    scrollHeight = 1_200

    await act(async () => {
      for (const observer of resizeObservers) {
        observer.trigger(1_200)
      }
    })
    await wait(0)

    expect(viewport.scrollTop).toBe(420)
  })

  it('renders reasoning text without a leading token space', () => {
    const { container } = render(<ReasoningHarness />)

    fireEvent.click(screen.getByRole('button', { name: /thinking/i }))

    expect(container.querySelector('[data-slot="aui_reasoning-text"]')?.textContent).toBe(
      'The user is asking what this file is.'
    )
  })

  it('groups consecutive reasoning parts under one thinking disclosure', () => {
    const { container } = render(<GroupedReasoningHarness />)

    const disclosures = container.querySelectorAll('[data-slot="aui_thinking-disclosure"]')
    expect(disclosures.length).toBe(1)

    fireEvent.click(disclosures[0].querySelector('button')!)

    const reasoningParts = container.querySelectorAll('[data-slot="aui_reasoning-text"]')
    expect(reasoningParts.length).toBe(2)
    expect(reasoningParts[0]?.textContent).toBe('First thought.')
    expect(reasoningParts[1]?.textContent).toBe('Second thought.')
  })

  it('renders live todo rows during a running turn', () => {
    const { container } = render(
      <TodoHarness
        message={assistantTodoMessage([
          { content: 'Gather ingredients', id: 'prep', status: 'completed' },
          { content: 'Boil water', id: 'boil', status: 'in_progress' }
        ])}
      />
    )

    const ui = within(container)

    expect(container.querySelector('[data-slot="aui_todo-hoisted"]')).toBeTruthy()
    expect(ui.getAllByText('Boil water').length).toBeGreaterThan(0)
    expect(ui.getByText('Gather ingredients')).toBeTruthy()
    expect(ui.queryByText(/pending/i)).toBeNull()
    expect(ui.queryByRole('button', { name: /todo/i })).toBeNull()
  })

  it('renders archived todos after turn completion regardless of pending state', () => {
    const first = render(
      <TodoHarness message={assistantTodoMessage([{ content: 'Boil water', id: 'boil', status: 'pending' }], false)} />
    )

    const ui = within(first.container)

    expect(ui.getAllByText('Boil water').length).toBeGreaterThan(0)

    first.unmount()

    const second = render(
      <TodoHarness
        message={assistantTodoMessage([{ content: 'Serve latte', id: 'serve', status: 'completed' }], false)}
      />
    )

    const archivedUi = within(second.container)

    expect(archivedUi.getAllByText('Serve latte').length).toBeGreaterThan(0)
  })

  it('hoists todo outside the thinking disclosure when reasoning is present', () => {
    const { container } = render(
      <TodoHarness
        message={assistantReasoningTodoMessage([
          { content: 'Buy oats', id: 'oats', status: 'completed' },
          { content: "Reply to Sam's email", id: 'email', status: 'in_progress' }
        ])}
      />
    )

    const todoPanel = container.querySelector('[data-slot="aui_todo-hoisted"]')
    const thinkingDisclosure = container.querySelector('[data-slot="aui_thinking-disclosure"]')

    expect(todoPanel).toBeTruthy()
    expect(thinkingDisclosure).toBeTruthy()
    expect(Boolean(thinkingDisclosure?.contains(todoPanel as Node))).toBe(false)
  })
})
