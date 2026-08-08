// Lists and blockquotes have chrome beside the text (markers, the quote
// border) whose side is driven by the box's CSS direction, which the
// unicode-bidi isolation never changes. These tests pin the split of
// responsibilities: block chrome carries a resolved dir so markers/borders
// follow Arabic/Hebrew text even when an item starts with an English brand or
// inline code, inline code carries dir="ltr" so it neither votes in that
// resolution nor reorders, and prose blocks carry the same resolved dir that
// `text-align:start` needs. jsdom does not resolve browser bidi visually, so
// the contract is asserted at the attribute level.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Thread } from '.'

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
vi.stubGlobal('CSS', { escape: (str: string) => str })

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

describe('block-level direction chrome', () => {
  it('lists resolve rtl so markers follow the sentence direction', async () => {
    render(<Harness text={'מקומות:\n\n1. חוף גורדון\n2. שוק הכרמל\n\n- פריט\n- item'} />)

    const item = await screen.findByText(/חוף גורדון/)

    expect(item.closest('ol')?.getAttribute('dir')).toBe('rtl')
    expect(item.closest('li')?.getAttribute('dir')).toBe('rtl')

    const bullet = await screen.findByText(/פריט/)

    expect(bullet.closest('ul')?.getAttribute('dir')).toBe('rtl')
    expect(bullet.closest('li')?.getAttribute('dir')).toBe('rtl')
  })

  it('blockquotes carry dir="auto" so the border follows the resolved direction', async () => {
    render(<Harness text={'> ציטוט קצר בעברית'} />)

    const quote = await screen.findByText(/ציטוט קצר/)

    expect(quote.closest('blockquote')?.getAttribute('dir')).toBe('rtl')
  })

  it('inline code carries dir="ltr" so it does not vote in resolved direction', async () => {
    render(<Harness text={'1. `npm install` מתקין תלויות'} />)

    const code = await screen.findByText('npm install')

    expect(code.tagName).toBe('CODE')
    expect(code.getAttribute('dir')).toBe('ltr')
    expect(code.closest('ol')?.getAttribute('dir')).toBe('rtl')
    expect(code.closest('li')?.getAttribute('dir')).toBe('rtl')
  })

  it('plain prose blocks carry resolved direction for text-align:start', async () => {
    render(<Harness text={'שלום לכולם'} />)

    const paragraph = await screen.findByText(/שלום לכולם/)

    expect(paragraph.closest('p')?.getAttribute('dir')).toBe('rtl')
  })

  it('brand-start Arabic bullet rows stay rtl instead of first-English LTR', async () => {
    render(
      <Harness
        text={
          '- **Alibaba** نزلت Qwen3.8-Max والمقلب الحلو إنك بتكلم الخبر ده\n' +
          '- **DeepSeek** نزلت V4 beta شغالة على الأسعار الصينية\n' +
          '- **Google عندها Gemini 3.5 + Gemini Omni + computer use في Flash** — بس برضه عندهم نزيف باحثين'
        }
      />
    )

    const alibaba = await screen.findByText('Alibaba')
    const deepseek = await screen.findByText('DeepSeek')
    const google = await screen.findByText(/Google عندها/)

    expect(alibaba.closest('ul')?.getAttribute('dir')).toBe('rtl')
    expect(alibaba.closest('li')?.getAttribute('dir')).toBe('rtl')
    expect(deepseek.closest('li')?.getAttribute('dir')).toBe('rtl')
    expect(google.closest('li')?.getAttribute('dir')).toBe('rtl')
  })
})
