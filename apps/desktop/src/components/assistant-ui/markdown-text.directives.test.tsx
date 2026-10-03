// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

import { MarkdownTextContent } from './markdown-text'

afterEach(cleanup)

function claimDemo() {
  return registry.register({
    id: 'test:directive-md',
    area: TRANSCRIPT_DIRECTIVE_AREA,
    source: 'plugin:test',
    data: {
      name: 'demo',
      render: ({ attrs }) => <div data-testid="demo-card">{attrs.p1 ?? 'demo'}</div>
    } satisfies TranscriptDirectiveContribution
  })
}

/**
 * End-to-end for the authored path: a directive paragraph has to survive
 * Streamdown's inline phase and reach the paragraph override as TEXT.
 *
 * GFM autolink literals are the hazard. An attribute that merely CONTAINS an
 * email address (or a bare www./http URL) makes remark split the paragraph
 * into text + <a> + text, and a paragraph with an element child was treated as
 * "not a directive" — so the card silently degraded to raw `::name{...}` in
 * front of the user. Attribute VALUES are data, never markup.
 */
describe('MarkdownTextContent transcript directives', () => {
  it('renders a directive whose attributes are plain text', () => {
    const dispose = claimDemo()

    try {
      render(<MarkdownTextContent isRunning={false} text='::demo{p1="Run the tests"}' />)

      expect(screen.getByTestId('demo-card').textContent).toBe('Run the tests')
      expect(screen.queryByText(/::demo\{/)).toBeNull()
    } finally {
      dispose()
    }
  })

  it('renders a directive whose attribute contains an email address', () => {
    const dispose = claimDemo()

    try {
      render(<MarkdownTextContent isRunning={false} text='::demo{p1="Why account btgrouptool@gmail.com is overloaded"}' />)

      expect(screen.getByTestId('demo-card').textContent).toBe('Why account btgrouptool@gmail.com is overloaded')
      expect(screen.queryByText(/::demo\{/)).toBeNull()
    } finally {
      dispose()
    }
  })

  it('renders a directive whose attribute contains a bare URL', () => {
    const dispose = claimDemo()

    try {
      render(<MarkdownTextContent isRunning={false} text='::demo{p1="Open www.example.com and compare"}' />)

      expect(screen.getByTestId('demo-card').textContent).toBe('Open www.example.com and compare')
      expect(screen.queryByText(/::demo\{/)).toBeNull()
    } finally {
      dispose()
    }
  })

  it('keeps an unclaimed directive as prose even when it holds an email', () => {
    render(<MarkdownTextContent isRunning={false} text='::nobody-home{p1="mail me at a@b.com"}' />)

    expect(screen.queryByTestId('demo-card')).toBeNull()
    expect(document.body.textContent).toContain('::nobody-home{')
  })
})
