import { describe, expect, it } from 'vitest'

import {
  createReplyComment,
  MAX_REPLY_COMMENTS,
  mergeReplyCommentsIntoDraft,
  normalizeReplyNote,
  normalizeReplyQuote,
  REPLY_COMMENT_NOTE_CHARS,
  REPLY_COMMENT_QUOTE_CHARS,
  serializeReplyComments,
  type ReplyComment
} from './reply-comments'

const comment = (quote: string, note: string): ReplyComment => {
  const created = createReplyComment(quote, note, `id-${quote.slice(0, 8)}`)

  if (!created) {
    throw new Error('expected a comment')
  }

  return created
}

describe('normalizeReplyQuote', () => {
  it('collapses whitespace and trims', () => {
    expect(normalizeReplyQuote('  hello\n\t  world  ')).toBe('hello world')
  })

  it('caps long passages with an ellipsis', () => {
    const long = 'w '.repeat(REPLY_COMMENT_QUOTE_CHARS)
    const normalized = normalizeReplyQuote(long)

    expect(normalized.endsWith('…')).toBe(true)
    expect(normalized.length).toBeLessThanOrEqual(REPLY_COMMENT_QUOTE_CHARS + 1)
  })

  it('keeps passages within budget untouched', () => {
    const quote = 'exact words'

    expect(normalizeReplyQuote(quote)).toBe(quote)
  })
})

describe('normalizeReplyNote', () => {
  it('collapses whitespace and caps length', () => {
    expect(normalizeReplyNote('  fix   this ')).toBe('fix this')

    const long = 'n'.repeat(REPLY_COMMENT_NOTE_CHARS + 50)

    expect(normalizeReplyNote(long).endsWith('…')).toBe(true)
  })
})

describe('createReplyComment', () => {
  it('returns null without a quoted passage', () => {
    expect(createReplyComment('   ', 'a note')).toBeNull()
  })

  it('accepts a bare quote with an empty note', () => {
    expect(createReplyComment('this part', '   ')).toMatchObject({ note: '', quote: 'this part' })
  })

  it('normalizes both fields', () => {
    expect(createReplyComment('  a\nb ', '  c\nd ')).toMatchObject({ note: 'c d', quote: 'a b' })
  })
})

describe('serializeReplyComments', () => {
  it('renders quote + labeled note per comment', () => {
    expect(serializeReplyComments([comment('the number is 42', 'this is wrong')])).toBe(
      '> the number is 42\nNote: this is wrong'
    )
  })

  it('renders bare quotes without a note line', () => {
    expect(serializeReplyComments([comment('just this', '')])).toBe('> just this')
  })

  it('joins several comments with a blank line', () => {
    const text = serializeReplyComments([comment('first', 'one'), comment('second', 'two')])

    expect(text).toBe('> first\nNote: one\n\n> second\nNote: two')
  })
})

describe('mergeReplyCommentsIntoDraft', () => {
  it('returns the draft untouched without comments', () => {
    expect(mergeReplyCommentsIntoDraft('hello', [])).toBe('hello')
  })

  it('freezes comment blocks ahead of the typed draft', () => {
    expect(mergeReplyCommentsIntoDraft('overall looks good', [comment('line 3', 'fix this')])).toBe(
      '> line 3\nNote: fix this\n\noverall looks good'
    )
  })

  it('sends bare comments when the draft is empty', () => {
    expect(mergeReplyCommentsIntoDraft('  ', [comment('line 3', 'fix this')])).toBe('> line 3\nNote: fix this')
  })
})

describe('budgets', () => {
  it('keeps the batch cap sane', () => {
    expect(MAX_REPLY_COMMENTS).toBeGreaterThan(0)
    expect(MAX_REPLY_COMMENTS).toBeLessThanOrEqual(20)
  })
})
