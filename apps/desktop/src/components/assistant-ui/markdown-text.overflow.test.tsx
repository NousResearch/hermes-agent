import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

// Regression coverage for the "workspace failed to render / Maximum call stack
// size exceeded" crash loop (#78000, #85295).
//
// Two independent recursions in the markdown pipeline can overflow the JS call
// stack while React is rendering:
//
//   1. Raw inline HTML — `rehype-raw` hands `<unk><unk>…` to parse5, and
//      `hast-util-from-parse5` recurses once per level of unclosed nesting. A
//      degenerating model emitting thousands of `<unk>` tokens as reasoning is
//      the reported trigger, and the payload persists to the session, so the
//      crash repeats on every reload.
//   2. Deeply nested block structure — `> > > …` recurses in mdast→hast, which
//      no HTML guard can prevent.
//
// The throw escaping to the workspace boundary is what blanks the whole app, so
// what matters is that every markdown surface stays up and keeps the text
// readable. These mount the real production component; a wrapper that silently
// stopped being applied has to fail here.
afterEach(cleanup)

// A stack overflow inside React's render logs through console.error; the test
// asserts on rendered output, not on the noise.
function renderQuietly(node: Parameters<typeof render>[0]) {
  const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

  try {
    return render(node)
  } finally {
    spy.mockRestore()
  }
}

const DEGENERATE_UNK = `Let${'<unk>'.repeat(16_383)}`
const DEEP_BLOCKQUOTE = `${'> '.repeat(10_000)}still readable`

describe('markdown surface survives stack-overflow content', () => {
  it.each([
    ['degenerate <unk> reasoning run', DEGENERATE_UNK],
    ['deeply nested blockquotes', DEEP_BLOCKQUOTE]
  ])('renders %s without throwing, and keeps it readable', (_label, text) => {
    const { container } = renderQuietly(<MarkdownTextContent isRunning={false} text={text} />)

    expect(container.textContent).toBeTruthy()
  })

  // The crash is a property of the CONTENT, not of which part carries it: the
  // same text arrives as an answer, as reasoning, or in a tool result, and all
  // three render through this component. Guarding only one of them is what let
  // the bug survive an earlier fix attempt.
  it('survives on the reasoning (disableArtifacts) path', () => {
    const { container } = renderQuietly(
      <MarkdownTextContent disableArtifacts isRunning={false} text={DEGENERATE_UNK} />
    )

    expect(container.textContent).toBeTruthy()
  })

  it('still renders while the message is streaming', () => {
    const { container } = renderQuietly(<MarkdownTextContent isRunning text={DEGENERATE_UNK} />)

    expect(container.textContent).toBeTruthy()
  })

  it('leaves ordinary markdown on the rich path', async () => {
    render(<MarkdownTextContent isRunning={false} text={'# Disk report\n\nC: is **full**'} />)

    // A real heading element — the rich renderer ran, rather than degrading
    // the whole message to plain text.
    expect(await screen.findByRole('heading', { name: 'Disk report' })).toBeTruthy()
    expect(screen.getByText('full')).toBeTruthy()
  })
})

describe('MarkdownTextContent overflow containment', () => {
  // #70451: wide content must never push a horizontal scrollbar onto the
  // transcript. Prose wraps; tables and fenced code keep content-sized layout
  // and scroll inside their own cards, so the thread itself stays one column
  // wide. The missing link was min-w-0 on the markdown container.
  it('keeps prose in the column while tables and code scroll inside their own cards', async () => {
    const longToken = 'x'.repeat(500)

    const text = [
      longToken,
      '',
      '```text',
      longToken,
      '```',
      '',
      '| Header with a long value | Other |',
      '| --- | --- |',
      `| ${longToken} | value |`
    ].join('\n')

    const { container } = render(<MarkdownTextContent isRunning={false} text={text} />)

    await waitFor(() => expect(container.querySelector('[data-slot="code-card"]')).toBeTruthy())

    const markdown = container.querySelector('.aui-md')!
    const codeCard = container.querySelector('[data-slot="code-card"]')!
    // The card's overlay scroller (ExpandableBlock's inner div).
    const codeScroller = codeCard.querySelector('.scrollbar-overlay')!
    const tableWrapper = container.querySelector('.aui-md-table')!

    expect(markdown.className).toContain('min-w-0')
    expect(codeScroller.className).toContain('overflow-x-auto')
    expect(codeCard.textContent).toContain(longToken)
    expect(tableWrapper.className).toContain('overflow-x-auto')
  })

  // jsdom has no layout engine, so the actual scrolling lives in styles.css
  // (`.aui-md .katex-display { overflow-x: auto; … }`). This pins the DOM
  // contract that rule hooks into: display math must arrive as a
  // `.katex-display` box nested inside the `.aui-md` container — the element
  // the stylesheet turns into the scroll container with the always-visible
  // slider. If the pipeline ever renders wide display math as something else
  // (inline `.katex`, raw source text), the overflow rule silently stops
  // applying and the formula's tail becomes unreachable again.
  it('nests long display equations in a KaTeX display block for the scroll CSS', async () => {
    const equation = `$$\n${'x+'.repeat(60)}x\n$$`

    const { container } = render(<MarkdownTextContent isRunning={false} text={equation} />)

    await waitFor(() => expect(container.querySelector('.katex-display')).toBeTruthy())

    const markdown = container.querySelector('.aui-md')!
    const display = container.querySelector('.katex-display')!

    expect(markdown.contains(display)).toBe(true)
    expect(display.querySelector(':scope > .katex')).not.toBeNull()
  })
})
