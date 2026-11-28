import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { preprocessMarkdown } from '@/lib/markdown-preprocess'

import { MarkdownTextContent } from './markdown-text'

afterEach(cleanup)

// CommonMark lets only a list starting at 1 interrupt a paragraph, so a reply
// numbering on from an earlier list rendered as one run-on paragraph.
it('renders an ordered list that continues numbering after a label line as a list', async () => {
  const { container } = render(<MarkdownTextContent text={'**Follow-ups**\n4. First\n5. Second'} />)
  await act(async () => Promise.resolve())

  const list = container.querySelector('ol')
  expect(list?.getAttribute('start')).toBe('4')
  expect([...(list?.querySelectorAll('li') ?? [])].map(li => li.textContent?.trim())).toEqual(['First', 'Second'])
})

it('leaves markdown that already renders correctly alone', () => {
  for (const markdown of [
    'Intro\n1. First\n2. Second',
    '1. First\ncontinued\n2. Second',
    '- Item\n  text\n3. Third',
    '## Heading\n4. Fourth',
    '> **Quote**\n> 4. Fourth',
    '```\nlabel\n4. not a list\n```'
  ]) {
    expect(preprocessMarkdown(markdown)).toBe(markdown)
  }
})
