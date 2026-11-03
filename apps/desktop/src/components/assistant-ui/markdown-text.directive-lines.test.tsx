// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

import { MarkdownTextContent } from './markdown-text'

// Render-level regression: remark-soft-breaks turns every single newline into
// a `<br>`, and a paragraph holding a `<br>` used to fail the text-only check,
// so directives written on consecutive lines rendered as raw `::name{…}` text.
afterEach(cleanup)

function registerDemo() {
  return registry.register({
    id: 'test:lines-demo',
    area: TRANSCRIPT_DIRECTIVE_AREA,
    source: 'plugin:test',
    data: {
      name: 'demo',
      render: ({ attrs }) => <div data-testid="demo-card">{attrs.label}</div>
    } satisfies TranscriptDirectiveContribution
  })
}

describe('transcript directives on consecutive lines', () => {
  it('renders every directive stacked one per line as a card', () => {
    const dispose = registerDemo()

    try {
      const { container } = render(
        <MarkdownTextContent
          isRunning={false}
          text={'::demo{label="one"}\n::demo{label="two"}\n::demo{label="three"}'}
        />
      )

      expect(screen.getAllByTestId('demo-card').map(card => card.textContent)).toEqual(['one', 'two', 'three'])
      expect(container.textContent).not.toContain('::demo')
    } finally {
      dispose()
    }
  })

  it('keeps the line breaks of prose sharing a paragraph with a directive', () => {
    const dispose = registerDemo()

    try {
      const { container } = render(
        <MarkdownTextContent isRunning={false} text={'First line\nsecond line\n::demo{label="card"}'} />
      )

      expect(screen.getByTestId('demo-card').textContent).toBe('card')
      expect(container.querySelector('p br')).not.toBeNull()
      expect(container.textContent).toContain('First line')
      expect(container.textContent).toContain('second line')
    } finally {
      dispose()
    }
  })
})
