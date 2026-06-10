// The chat view pairs useExternalMessageConverter output with the source
// ChatMessage array by index to derive branch parents, and the incremental
// runtime depends on converted messages keeping referential identity across
// busy flips and unrelated updates. These tests pin both invariants, plus
// the action bar's copy-through-primitive behavior.
import { AssistantRuntimeProvider, useExternalMessageConverter, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessage } from '@assistant-ui/react'
import { act, render, renderHook, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Thread } from '@/components/assistant-ui/thread'
import type { ChatMessage } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
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

const user = (id: string, text: string): ChatMessage => ({
  id,
  role: 'user',
  parts: [{ type: 'text', text }],
  timestamp: 1718000000
})

const assistant = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role: 'assistant',
  parts: [{ type: 'text', text }],
  timestamp: 1718000001,
  ...extra
})

describe('useExternalMessageConverter swap', () => {
  const convert = (messages: ChatMessage[], busy: boolean) =>
    renderHook(
      ({ msgs, running }: { msgs: ChatMessage[]; running: boolean }) =>
        useExternalMessageConverter<ChatMessage>({
          callback: toRuntimeMessage,
          messages: msgs,
          isRunning: running,
          joinStrategy: 'none'
        }),
      { initialProps: { msgs: messages, running: busy } }
    )

  it('stays 1:1 and id-aligned across roles, hidden messages, and branch groups', () => {
    const messages: ChatMessage[] = [
      user('u1', 'hello'),
      assistant('a1', 'first answer', { branchGroupId: 'g1' }),
      assistant('a2', 'regenerated answer', { branchGroupId: 'g1' }),
      { ...user('u2', 'hidden context'), hidden: true } as ChatMessage,
      { id: 's1', role: 'system', parts: [{ type: 'text', text: 'slash:/help\nok' }], timestamp: 1718000002 },
      assistant('a3', 'streaming', { pending: true })
    ]

    const { result } = convert(messages, true)

    expect(result.current).toHaveLength(messages.length)
    expect(result.current.map(m => m.id)).toEqual(messages.map(m => m.id))
    expect(result.current.map(m => m.role)).toEqual(['user', 'assistant', 'assistant', 'user', 'system', 'assistant'])
    expect(result.current[5]?.status?.type).toBe('running')
    expect(result.current[1]?.status?.type).toBe('complete')
  })

  it('keeps message identity stable when busy flips', () => {
    const messages = [user('u1', 'hello'), assistant('a1', 'done')]
    const { result, rerender } = convert(messages, true)
    const before = [...result.current]

    rerender({ msgs: messages, running: false })

    expect(result.current[0]).toBe(before[0])
    expect(result.current[1]).toBe(before[1])
  })

  it('only re-converts the message object that changed', () => {
    const stable = user('u1', 'hello')
    const v1 = assistant('a1', 'partial', { pending: true })
    const { result, rerender } = convert([stable, v1], true)
    const before = [...result.current]

    const v2 = assistant('a1', 'partial plus more', { pending: true })

    rerender({ msgs: [stable, v2], running: true })

    expect(result.current[0]).toBe(before[0])
    expect(result.current[1]).not.toBe(before[1])
    expect(result.current[1]?.content.map(part => (part.type === 'text' ? part.text : part.type))).toEqual([
      'partial plus more'
    ])
  })
})

const doneAssistant = (text: string): ThreadMessage =>
  ({
    id: 'assistant-copy-1',
    role: 'assistant',
    content: [
      { type: 'text', text },
      { type: 'text', text: 'second paragraph' }
    ],
    createdAt: new Date(1718000001000),
    status: { type: 'complete', reason: 'stop' },
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  }) as ThreadMessage

function CopyHarness() {
  const [messages] = useState<ThreadMessage[]>([doneAssistant('copy me')])
  const runtime = useExternalStoreRuntime<ThreadMessage>({ messages, isRunning: false, onNew: async () => {} })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('ActionBarPrimitive.Copy swap', () => {
  beforeEach(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) }
    })
  })

  it('copies the message text through the primitive', async () => {
    const { container } = render(<CopyHarness />)
    const bar = container.querySelector('[data-slot="aui_msg-actions"]')
    const button = bar?.querySelector('button')

    expect(button).toBeTruthy()

    await act(async () => {
      button?.click()
    })

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('copy me\n\nsecond paragraph')

    await waitFor(() => expect(button?.getAttribute('data-copied')).toBe('true'))
  })
})
