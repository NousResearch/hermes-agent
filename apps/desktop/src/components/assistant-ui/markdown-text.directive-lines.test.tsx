// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

import { MarkdownTextContent } from './markdown-text'

// Render-level: mounts the real markdown surface (preprocess, remark plugins,
// sanitize, components) and asserts on the DOM. A directive is lifted into its
// own element before remark-soft-breaks runs, so what sits next to it in the
// paragraph — a line break, bold, a link — must not hide it any more.
let dispose: () => void

beforeEach(() => {
  dispose = registry.register({
    id: 'test:render-demo',
    area: TRANSCRIPT_DIRECTIVE_AREA,
    source: 'plugin:test',
    data: {
      name: 'demo',
      render: ({ attrs }) => <div data-testid="demo-card">{attrs.label}</div>
    } satisfies TranscriptDirectiveContribution
  })
})

afterEach(() => {
  dispose()
  cleanup()
})

const cards = () => screen.queryAllByTestId('demo-card').map(card => card.textContent)
const mount = (text: string) => render(<MarkdownTextContent isRunning={false} text={text} />).container

describe('transcript directives in rendered markdown', () => {
  it('renders directives stacked one per line, each as a card', () => {
    const container = mount('::demo{label="one"}\n::demo{label="two"}\n::demo{label="three"}')

    expect(cards()).toEqual(['one', 'two', 'three'])
    expect(container.textContent).not.toContain('::demo')
  })

  it('renders a directive under a line of prose and keeps that prose and its line breaks', () => {
    const container = mount('First line\nsecond line\n::demo{label="card"}')

    expect(cards()).toEqual(['card'])
    expect(container.querySelector('p br')).not.toBeNull()
    expect(container.textContent).toContain('First line')
    expect(container.textContent).toContain('second line')
    expect(container.textContent).not.toContain('::demo')
  })

  it('renders a directive in a paragraph that also holds inline markup', () => {
    const container = mount('Here is **the** [doc](https://example.com): ::demo{label="card"}')

    expect(cards()).toEqual(['card'])
    expect(container.querySelector('p [data-streamdown="strong"]')?.textContent).toBe('the')
    expect(container.querySelector('p a')).not.toBeNull()
  })

  it('leaves an unclaimed directive in its sentence as the text it was written as', () => {
    const container = mount('Use ::nobody{x="1"} here')

    expect(cards()).toEqual([])
    expect(container.querySelector('p')?.textContent).toBe('Use ::nobody{x="1"} here')
  })

  it('does not lift a directive glued to the markup before it', () => {
    const container = mount('**a**::demo{label="glued"}')

    expect(cards()).toEqual([])
    expect(container.textContent).toContain('::demo{label="glued"}')
  })

  it('leaves a directive in a heading as text', () => {
    const container = mount('# Title ::demo{label="h"}')

    expect(cards()).toEqual([])
    expect(container.querySelector('h1')?.textContent).toContain('::demo{label="h"}')
  })
})
