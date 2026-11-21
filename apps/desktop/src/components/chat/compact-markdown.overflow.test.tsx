import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { CompactMarkdown } from './compact-markdown'

afterEach(cleanup)

// #70451: CompactMarkdown renders tool detail bodies, where a fenced log line
// used to grow a nested horizontal scrollbar. Fences soft-wrap; tables keep
// content-sized columns and scroll inside their own wrapper, like code cards.
describe('CompactMarkdown overflow containment', () => {
  it('soft-wraps fenced code instead of scrolling sideways', () => {
    const longToken = 'x'.repeat(500)

    const { container } = render(<CompactMarkdown text={`\`\`\`text\n${longToken}\n\`\`\``} />)

    const pre = container.querySelector('pre')!
    expect(pre).toBeTruthy()
    expect(pre.className).toContain('overflow-x-hidden')
    expect(pre.className).toContain('whitespace-pre-wrap')
    expect(pre.className).not.toContain('overflow-x-auto')
  })

  // #145797: the body's `wrap-anywhere` must not reach table cells, or auto
  // layout shrinks short columns to one letter per line.
  it('wraps table cells between words, not letters', () => {
    const { container } = render(
      <CompactMarkdown text={'| Label | Detail |\n| --- | --- |\n| Upstream files touched | Long prose detail |'} />
    )

    const cells = Array.from(container.querySelectorAll('td'))

    expect(cells).toHaveLength(2)

    for (const cell of cells) {
      expect(cell.className).toContain('wrap-break-word')
      expect(cell.className).not.toContain('wrap-anywhere')
    }
  })
})
