// A markdown table's column order is its box `direction`, which the
// unicode-bidi:plaintext rules that own per-block text direction never touch.
// The fix hangs dir="auto" on the <table> so the browser resolves column order
// from the cells' content, and aligns headers/cells with text-align:start
// (text-start, not a pinned text-left) so they follow that resolved direction.
// jsdom does not resolve dir="auto" or apply the stylesheet, so the contract is
// asserted at the attribute/class level.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Thread } from './thread'

const createdAt = new Date('2026-06-01T00:00:00.000Z')

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

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'hi' }],
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

function Harness({ text }: { text: string }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistantMessage(text)],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

const HEBREW_TABLE = [
  '| שם פריט | כמות | מחיר |',
  '| --- | --- | --- |',
  '| תפוחים | 12 | 18 |',
  '| בננות | 8 | 14 |'
].join('\n')

const ENGLISH_TABLE = ['| Item | Qty | Price |', '| --- | --- | --- |', '| Apples | 12 | 4.50 |'].join('\n')

afterEach(cleanup)

describe('markdown table direction', () => {
  it('a Hebrew table carries dir="auto" so the browser resolves column order from content', async () => {
    render(<Harness text={HEBREW_TABLE} />)

    const cell = await screen.findByText('תפוחים')

    expect(cell.closest('table')?.getAttribute('dir')).toBe('auto')
  })

  it('header cells use logical alignment so headers follow the resolved direction', async () => {
    render(<Harness text={HEBREW_TABLE} />)

    const header = await screen.findByText('שם פריט')

    expect(header.tagName).toBe('TH')
    expect(header.className).toContain('text-start')
    expect(header.className).not.toContain('text-left')
  })

  it('the dir="auto" hook is content-driven (not pinned RTL), so an English table carries it too', async () => {
    render(<Harness text={ENGLISH_TABLE} />)

    const cell = await screen.findByText('Apples')

    expect(cell.closest('table')?.getAttribute('dir')).toBe('auto')
  })
})
