import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

afterEach(cleanup)

describe('MarkdownTextContent overflow containment', () => {
  it('wraps prose and tables while preserving fenced-code scrolling', async () => {
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
    const codePre = container.querySelector('[data-slot="code-card"] pre')!
    const codeScroller = container.querySelector('[data-slot="code-card"] .scrollbar-overlay')!
    const tableWrapper = container.querySelector('.aui-md-table')!
    const table = tableWrapper.querySelector('table')!
    const header = table.querySelector('th')!
    const cell = table.querySelector('td')!

    expect(markdown.className).toContain('min-w-0')
    expect(codePre.querySelector('code')?.className).toContain('whitespace-pre')
    expect(codeScroller.className).toContain('overflow-x-auto')
    expect(tableWrapper.className).toContain('overflow-hidden')
    expect(table.className).toContain('table-fixed')
    expect(header.className).toContain('wrap-anywhere')
    expect(cell.className).toContain('wrap-anywhere')
  })
})
