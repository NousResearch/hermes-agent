import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Thread } from '.'

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

// jsdom returns 0 for offset*; some layout code reads those to size the
// viewport. Fall through to client* (which tests can override) or a sane
// default so message rows render with non-zero dimensions.
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

function assistantReasoningMessage(text: string, running = false): ThreadMessage {
  return {
    id: 'assistant-reasoning-1',
    role: 'assistant',
    content: [{ type: 'reasoning', text }],
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

function assistantSeparatedReasoningMessage(): ThreadMessage {
  return {
    id: 'assistant-reasoning-separated-1',
    role: 'assistant',
    content: [
      { type: 'reasoning', text: ' Complete first thought.', status: { type: 'complete' } },
      { type: 'text', text: 'Interim answer.' },
      { type: 'reasoning', text: ' Streaming second thought.', status: { type: 'running' } }
    ],
    status: { type: 'running' },
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

function assistantCommentaryMessage(): ThreadMessage {
  return {
    id: 'assistant-commentary-1',
    role: 'assistant',
    content: [
      { type: 'reasoning', text: ' Planning the analysis.', status: { type: 'complete' } },
      { type: 'data', name: 'commentary', data: { text: 'Reading the screenshot first.' } },
      { type: 'text', text: 'The total is $42.' }
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

function assistantImageMessage(running = false): ThreadMessage {
  return {
    id: `assistant-image-${running ? 'running' : 'done'}`,
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'image-1',
        toolName: 'image_generate',
        args: { prompt: 'draw a cat' },
        argsText: JSON.stringify({ prompt: 'draw a cat' }),
        ...(running ? {} : { result: { image: 'https://cdn.example/cat.png', success: true } })
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

function RunningMessageHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: true,
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

function RunningReasoningHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantReasoningMessage('```ts\nconst answer = 42\n', true)],
    isRunning: true,
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

function CommentaryHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantCommentaryMessage()],
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

function DismissibleErrorHarness({ onDismissError }: { onDismissError: (messageId: string) => void }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantErrorMessage('OpenRouter rejected the request (403).')],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread onDismissError={onDismissError} />
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

  it('omits the dismiss control when no onDismissError handler is supplied', () => {
    render(<MessageHarness message={assistantErrorMessage('OpenRouter rejected the request (403).')} />)

    expect(screen.queryByRole('button', { name: 'Dismiss error' })).toBeNull()
  })

  it('invokes onDismissError with the errored message id when the dismiss control is clicked', () => {
    const onDismissError = vi.fn()
    render(<DismissibleErrorHarness onDismissError={onDismissError} />)

    const dismiss = screen.getByRole('button', { name: 'Dismiss error' })
    fireEvent.click(dismiss)

    expect(onDismissError).toHaveBeenCalledTimes(1)
    expect(onDismissError).toHaveBeenCalledWith('assistant-error-1')
  })

  // Scroll behavior (follow-at-bottom, escape-on-scroll-up, re-engage) is owned
  // by the use-stick-to-bottom library and covered by its own test suite. We
  // don't re-assert its scrollTop mechanics here — doing so in jsdom (no real
  // layout, spring animation via rAF) only produces brittle change-detector
  // tests. The rendering/streaming-content tests below remain the contract.

  it('renders an incomplete streaming fenced code block as a code card', async () => {
    const { container } = render(<RunningMessageHarness message={assistantMessage('```ts\nconst answer = 42\n')} />)

    await waitFor(() => {
      expect(container.querySelector('[data-slot="code-card"]')).toBeTruthy()
    })

    expect(container.textContent).toContain('const answer = 42')
    expect(container.textContent).not.toContain('```ts')
  })

  it('renders an incomplete streaming reasoning fenced code block as a code card', async () => {
    const { container } = render(<RunningReasoningHarness />)
    const ui = within(container)
    const thinkingToggle = ui.getByRole('button', { name: /thinking/i })

    if (thinkingToggle.getAttribute('aria-expanded') !== 'true') {
      fireEvent.click(thinkingToggle)
    }

    await waitFor(() => {
      expect(container.querySelector('[data-slot="code-card"]')).toBeTruthy()
    })

    await waitFor(() => {
      expect(container.querySelector('[data-slot="aui_reasoning-text"]')?.textContent).toContain('const answer = 42')
    })
    expect(container.textContent).not.toContain('```ts')
  })

  it('renders reasoning text without a leading token space', () => {
    const { container } = render(<ReasoningHarness />)
    const ui = within(container)

    fireEvent.click(ui.getByRole('button', { name: /thinking/i }))

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

  it('renders Codex commentary in a separate Working disclosure, not under Thinking', () => {
    const { container } = render(<CommentaryHarness />)
    const ui = within(container)

    const thinkingDisclosures = container.querySelectorAll('[data-slot="aui_thinking-disclosure"]')
    const commentaryDisclosures = container.querySelectorAll('[data-slot="aui_commentary-disclosure"]')
    expect(thinkingDisclosures.length).toBe(1)
    expect(commentaryDisclosures.length).toBe(1)

    // Working lane opens to the narration text.
    fireEvent.click(ui.getByRole('button', { name: /working/i }))
    expect(container.querySelector('[data-slot="aui_commentary-text"]')?.textContent).toBe(
      'Reading the screenshot first.'
    )

    // Thinking lane holds only the genuine reasoning.
    fireEvent.click(ui.getByRole('button', { name: /thinking/i }))
    const reasoningText = container.querySelector('[data-slot="aui_reasoning-text"]')?.textContent ?? ''
    expect(reasoningText).toBe('Planning the analysis.')
    expect(reasoningText).not.toContain('screenshot')

    // Final answer renders as normal chat text.
    expect(container.textContent).toContain('The total is $42.')
  })

  it('does not reopen an earlier completed thinking group when a later group is running', () => {
    const { container } = render(<RunningMessageHarness message={assistantSeparatedReasoningMessage()} />)

    const disclosures = container.querySelectorAll('[data-slot="aui_thinking-disclosure"]')
    expect(disclosures.length).toBe(2)

    expect(disclosures[0].querySelector('button')?.getAttribute('aria-expanded')).toBe('false')
    expect(disclosures[1].querySelector('button')?.getAttribute('aria-expanded')).toBe('true')
    expect(container.textContent).not.toContain('Complete first thought.')
    expect(container.textContent).toContain('Interim answer.')
  })

  it('does not render an inline todo panel — todos live in the composer status stack', () => {
    const { container } = render(
      <TodoHarness
        message={assistantTodoMessage([
          { content: 'Gather ingredients', id: 'prep', status: 'completed' },
          { content: 'Boil water', id: 'boil', status: 'in_progress' }
        ])}
      />
    )

    expect(container.querySelector('[data-slot="aui_todo-hoisted"]')).toBeNull()
  })

  it('renders completed image generation results in the tool slot', async () => {
    const { container } = render(<MessageHarness message={assistantImageMessage()} />)

    await waitFor(() => {
      expect(screen.getByRole('img', { name: 'Generated image' }).getAttribute('src')).toBe(
        'https://cdn.example/cat.png'
      )
    })
    expect(container.querySelector('[data-slot="aui_generated-image"]')).toBeTruthy()
    expect(screen.queryByRole('status', { name: /rendering image/i })).toBeNull()
  })
})
