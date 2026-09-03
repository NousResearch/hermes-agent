import { cleanup, render, screen } from '@testing-library/react'
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
  it.each([
    ['reasoning (disableArtifacts)', { disableArtifacts: true }],
    ['assistant answer', {}]
  ])('survives on the %s path', (_label, surfaceProps) => {
    const { container } = renderQuietly(
      <MarkdownTextContent isRunning={false} text={DEGENERATE_UNK} {...surfaceProps} />
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
