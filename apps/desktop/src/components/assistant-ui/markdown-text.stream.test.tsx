import { cleanup, render, waitFor } from '@testing-library/react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// Streaming render path contracts: what runs at urgent priority per flush, and
// that the streamed DOM is the DOM a fresh render of the same text produces.

const { codeToHtml } = vi.hoisted(() => ({
  codeToHtml: vi.fn((code: string) => {
    const escaped = String(code).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

    return `<pre class="shiki"><code>${escaped}</code></pre>`
  })
}))

// jsdom cannot run shiki's wasm engine; a deterministic highlighter keeps the
// settled DOM comparable and makes "highlighted" observable.
vi.mock('shiki', () => ({
  bundledLanguages: { python: 'python-loader', text: 'text-loader', ts: 'ts-loader' },
  getSingletonHighlighter: vi.fn(async () => ({
    codeToHtml: (code: string) => codeToHtml(code),
    getLoadedLanguages: () => ['text', 'ts', 'python'],
    loadLanguage: vi.fn(async () => undefined)
  }))
}))

vi.mock('shiki/engine/oniguruma', () => ({
  createOnigurumaEngine: vi.fn(() => ({}) as never)
}))

vi.mock('@/lib/markdown-preprocess', async importActual => {
  const actual = await importActual<typeof MarkdownPreprocess>()

  return { ...actual, preprocessMarkdown: vi.fn(actual.preprocessMarkdown) }
})

// One remark transform per Block parse: the cost a re-rendered Block pays.
const { blockParses } = vi.hoisted(() => ({ blockParses: { count: 0 } }))

vi.mock('@/lib/remark-soft-breaks', async importActual => {
  const actual = await importActual<typeof RemarkSoftBreaks>()

  return {
    ...actual,
    remarkSoftBreaks: () => {
      const transform = actual.remarkSoftBreaks()

      return (tree: Parameters<typeof transform>[0]) => {
        blockParses.count += 1
        transform(tree)
      }
    }
  }
})

import { I18nProvider } from '@/i18n'
import type * as MarkdownPreprocess from '@/lib/markdown-preprocess'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import type * as RemarkSoftBreaks from '@/lib/remark-soft-breaks'

import { MarkdownTextContent } from './markdown-text'

type ActGlobal = typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }

const preprocessSpy = vi.mocked(preprocessMarkdown)

const CORPUS = [
  '<thinking>private chain of thought</thinking># Release notes',
  '',
  'Intro with **bold**, `inline code`, a [link](https://example.com) and $x^2$ math [1].',
  '',
  '- item one',
  '- item two',
  '  - nested',
  '',
  '1. first',
  '2. second',
  '',
  '| col a | col b |',
  '|---|---|',
  '| 1 | 2 |',
  '',
  '$$',
  '\\int_0^1 x\\,dx = \\tfrac12',
  '$$',
  '',
  '```python',
  'def f(x):',
  '    return x * 2',
  '```',
  '',
  'Paragraph between fences.',
  '',
  '```ts',
  'const answer: number = 42',
  'console.log(answer)',
  '```',
  '',
  '## Sources',
  '',
  '[1] https://example.com/source',
  ''
].join('\n')

function pause(ms = 20): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>()
  setTimeout(resolve, ms)

  return promise
}

async function settled(container: HTMLElement, text: string) {
  // The deferred render has committed once the last visible words are painted.
  const tail = text.trim().split(/\s+/).at(-1) ?? ''
  await waitFor(() => expect(container.textContent).toContain(tail.replace(/[`*$|[\]]/g, '')))
}

async function freshHtml(text: string, isRunning: boolean): Promise<string> {
  const { container, unmount } = render(<MarkdownTextContent isRunning={isRunning} text={text} />)

  if (!isRunning) {
    await waitFor(() => expect(container.querySelectorAll('[data-testid="shiki-container"]').length).toBe(2))
  }

  await pause()
  const html = container.innerHTML
  unmount()

  return html
}

beforeAll(async () => {
  // Land the lazily imported code plugin (and the shiki chunk) once, so every
  // render below sees the same plugin table.
  ;(globalThis as ActGlobal).IS_REACT_ACT_ENVIRONMENT = false
  await freshHtml(CORPUS, false)
})

beforeEach(() => {
  ;(globalThis as ActGlobal).IS_REACT_ACT_ENVIRONMENT = false
  preprocessSpy.mockClear()
})

afterEach(() => {
  cleanup()
  ;(globalThis as ActGlobal).IS_REACT_ACT_ENVIRONMENT = true
})

describe('MarkdownTextContent streaming render path', () => {
  it('keeps the full-text preprocess off the urgent render of a flush', async () => {
    const before = CORPUS.slice(0, 400)
    const after = CORPUS.slice(0, 480)
    // A bare root, not RTL's act()-wrapped rerender: act drains the deferred
    // lane inside the same call, which would hide which priority did the work.
    const container = window.document.body.appendChild(window.document.createElement('div'))
    const root = createRoot(container)
    flushSync(() => root.render(<MarkdownTextContent isRunning text={before} />))
    await settled(container, before)
    preprocessSpy.mockClear()

    // A flush lands as an urgent (sync-lane) update; flushSync returns after
    // committing that lane only.
    flushSync(() => root.render(<MarkdownTextContent isRunning text={after} />))

    expect(preprocessSpy.mock.calls.some(([text]) => text === after)).toBe(false)

    // ...and the deferred render still delivers it.
    await waitFor(() => expect(preprocessSpy.mock.calls.some(([text]) => text === after)).toBe(true))
    root.unmount()
    container.remove()
  })

  it('paints, at every streaming cut and after the stream ends, exactly what a fresh render paints', async () => {
    const { container, rerender } = render(<MarkdownTextContent isRunning text="" />)

    // Cuts land mid-fence, mid-table, mid-math and mid-reasoning-tag.
    for (const end of [20, 170, 260, 330, 380, 420, 470, CORPUS.length]) {
      const text = CORPUS.slice(0, end)
      rerender(<MarkdownTextContent isRunning text={text} />)
      await pause()
      const streamed = container.innerHTML
      expect(streamed).toBe(await freshHtml(text, true))
    }

    rerender(<MarkdownTextContent isRunning={false} text={CORPUS} />)
    await waitFor(() => expect(container.querySelectorAll('[data-testid="shiki-container"]').length).toBe(2))
    await pause()

    expect(container.innerHTML).toBe(await freshHtml(CORPUS, false))
    expect(container.textContent).not.toContain('private chain of thought')
  })

  it('recovers a listing that fell back while streaming without reparsing settled markdown', async () => {
    const text = 'Before the listing.\n\n```listing\n{"address":"123 Main St","url":"https://example.com/listing"}\n```'

    const content = (running: boolean) => (
      <I18nProvider><MarkdownTextContent isRunning={running} text={text} /></I18nProvider>
    )

    // Warm the lazy renderer so the running mount actually reaches its fallback.
    const warm = render(content(false))
    await waitFor(() => expect(warm.container.querySelector('[data-slot="aui_listing-card"]')).not.toBeNull())
    warm.unmount()

    const { container, rerender } = render(content(true))
    await settled(container, 'example.com/listing')
    expect(container.querySelector('[data-slot="aui_listing-card"]')).toBeNull()
    blockParses.count = 0

    rerender(content(false))
    await waitFor(() => expect(container.querySelector('[data-slot="aui_listing-card"]')).not.toBeNull())
    expect(container.querySelector('a')?.getAttribute('href')).toBe('https://example.com/listing')
    expect(blockParses.count).toBe(0)
  })

  it('ends a stream without re-parsing settled blocks, and highlights the code that streamed plain', async () => {
    // The reply ends inside its last block — a code fence — so the tail is the
    // block that streamed as plain code and must highlight at stream end.
    const text = `${CORPUS}\nOne more fence:\n\n\`\`\`ts\nconst tail = true\n\`\`\``
    const { container, rerender } = render(<MarkdownTextContent isRunning text={text} />)
    await settled(container, 'tail = true')
    await pause()

    expect(container.querySelector('[data-testid="shiki-container"]')).toBeNull()
    const cards = [...container.querySelectorAll('[data-slot="code-card"]')]
    expect(cards).toHaveLength(3)
    blockParses.count = 0

    rerender(<MarkdownTextContent isRunning={false} text={text} />)
    await waitFor(() => expect(container.querySelectorAll('[data-testid="shiki-container"]').length).toBe(3))

    // No Block re-parses its unchanged markdown, and every code card is the
    // same DOM node: highlighting swapped in place instead of remounting.
    expect(blockParses.count).toBe(0)
    expect([...container.querySelectorAll('[data-slot="code-card"]')]).toEqual(cards)
    expect(cards.at(-1)?.querySelector('[data-testid="shiki-container"]')?.textContent).toContain('const tail = true')
  })
})
