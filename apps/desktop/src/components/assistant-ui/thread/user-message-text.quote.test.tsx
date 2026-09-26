import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { QuotedPassage, splitLeadingQuote, UserMessageText } from './user-message-text'

describe('splitLeadingQuote', () => {
  it('separates the leading passage from the reply', () => {
    expect(splitLeadingQuote('> quoted\n>\n> more\n\nmy words here')).toEqual({
      body: 'my words here',
      lines: ['quoted', '', 'more']
    })
  })

  it('keeps a quote-free message whole', () => {
    expect(splitLeadingQuote('just my words')).toEqual({ body: 'just my words', lines: null })
  })

  it('leaves a ">" that is not the opening run alone', () => {
    expect(splitLeadingQuote('first line\n> second line')).toEqual({ body: 'first line\n> second line', lines: null })
  })

  it('handles a passage with no reply under it', () => {
    expect(splitLeadingQuote('> quoted only')).toEqual({ body: '', lines: ['quoted only'] })
  })
})

describe('QuotedPassage', () => {
  it('labels the passage so a reply reads as a reply', () => {
    const { container } = render(<QuotedPassage lines={['the quoted answer']} />)

    expect(container.querySelector('[data-slot="aui_user-quote"]')?.textContent).toContain('Follow-up')
    expect(container.querySelector('[data-slot="aui_user-quote-text"]')?.textContent).toBe('the quoted answer')
  })
})

describe('UserMessageText — quoted passages', () => {
  it('renders a follow-up block as a quote, not as literal "> " text', () => {
    const { container } = render(<UserMessageText text={'> the quoted passage\n\nwhat about it?'} />)

    const quote = container.querySelector('[data-slot="aui_user-quote"]')
    const passage = container.querySelector('[data-slot="aui_user-quote-text"]')

    // Labelled, so a reply reads as a reply instead of as the reader's own text.
    expect(quote?.textContent).toContain('Follow-up')
    expect(passage?.textContent).toBe('the quoted passage')
    expect(container.textContent).not.toContain('> the quoted passage')
    expect(container.textContent).toContain('what about it?')
  })

  it('keeps a multi-paragraph passage in one quote block', () => {
    const { container } = render(<UserMessageText text={'> para one\n>\n> para two\n\nanswer me'} />)

    const quotes = container.querySelectorAll('[data-slot="aui_user-quote"]')

    expect(quotes).toHaveLength(1)
    expect(container.querySelector('[data-slot="aui_user-quote-text"]')?.textContent).toBe('para one\n\npara two')
  })

  it('leaves a message without a quote alone', () => {
    const { container } = render(<UserMessageText text={'plain text > inside a sentence'} />)

    expect(container.querySelector('[data-slot="aui_user-quote"]')).toBeNull()
    expect(container.textContent).toBe('plain text > inside a sentence')
  })

  it('still renders fenced code that contains a quote-like line', () => {
    const { container } = render(<UserMessageText text={'```\n> quoted inside code\n```'} />)

    expect(container.querySelector('[data-slot="aui_user-fence"]')?.textContent).toContain('> quoted inside code')
    expect(container.querySelector('[data-slot="aui_user-quote"]')).toBeNull()
  })

  it('elides a passage past the preview budget and keeps the rest on hover', () => {
    const long = 'z'.repeat(600)
    const { container } = render(<UserMessageText text={`> ${long}\n\nwhy?`} />)

    const passage = container.querySelector('[data-slot="aui_user-quote-text"]')

    expect(passage?.textContent?.endsWith('…')).toBe(true)
    expect(passage?.textContent?.length).toBeLessThan(long.length)
    // The whole passage is one hover away — the message itself still carries it.
    expect(passage?.getAttribute('title')).toBe(long)
  })
})
