// Message text must resolve its direction from its own prose so RTL
// scripts (Hebrew, Arabic) render right-aligned and correctly ordered per
// block, while code stays LTR. Code spans don't get a vote: a technical
// RTL message often *starts* with a command, which would flip plain
// first-strong detection to LTR. jsdom does not compute visual direction,
// so these tests pin the contract that drives it in the browser: prose
// blocks carry the resolved dir attribute, code blocks never carry one.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Thread } from './thread'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

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

function userMessage(text: string): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(text: string): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text }],
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

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('message text direction', () => {
  it('user message text resolves direction from content while fences stay LTR', async () => {
    render(<Harness messages={[userMessage('שלום עולם\n```\nnpm run dev\n```')]} />)

    const text = await screen.findByText(/שלום עולם/)

    expect(text.closest('[dir]')?.getAttribute('dir')).toBe('rtl')

    const code = screen.getByText(/npm run dev/)

    expect(code.closest('pre')).not.toBeNull()
    expect(code.closest('[dir]')).toBeNull()
  })

  it('user message starting with inline code still follows its prose', async () => {
    render(<Harness messages={[userMessage('`./scripts/run.sh -v` מה הפקודה הזאת עושה?')]} />)

    const text = await screen.findByText(/מה הפקודה הזאת עושה/)

    expect(text.closest('[dir]')?.getAttribute('dir')).toBe('rtl')
  })

  it('assistant prose blocks resolve direction per block', async () => {
    render(<Harness messages={[userMessage('hi'), assistantMessage('שלום לכולם\n\n- פריט ראשון\n- second item')]} />)

    const paragraph = await screen.findByText(/שלום לכולם/)

    expect(paragraph.closest('p')?.getAttribute('dir')).toBe('rtl')

    const item = await screen.findByText(/פריט ראשון/)

    expect(item.closest('li')?.getAttribute('dir')).toBe('rtl')
    expect(item.closest('ul')?.getAttribute('dir')).toBe('rtl')
  })

  it('assistant paragraphs starting with inline code follow their prose', async () => {
    render(
      <Harness
        messages={[
          userMessage('hi'),
          assistantMessage('`npm run dev` מפעיל את סביבת הפיתוח.\n\n`npm run dev` starts the dev environment.')
        ]}
      />
    )

    const rtl = await screen.findByText(/מפעיל את סביבת הפיתוח/)

    expect(rtl.closest('p')?.getAttribute('dir')).toBe('rtl')

    const ltr = await screen.findByText(/starts the dev environment/)

    expect(ltr.closest('p')?.getAttribute('dir')).toBe('ltr')
  })
})
