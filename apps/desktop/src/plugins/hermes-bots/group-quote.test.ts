import { describe, expect, it } from 'vitest'

import { GROUP_QUOTE_TEXT_CHARS, quoteFromMessage } from './group-quote'
import type { GroupMessage } from './types'

const message = (text: string): GroupMessage => ({ at: 1, from: { kind: 'member', name: 'builder' }, text })

describe('quoteFromMessage', () => {
  it('snapshots the speaker, the line and when it was taken', () => {
    expect(quoteFromMessage(message('totals are 5'), 'Builder', 42)).toEqual({
      at: 42,
      from: 'Builder',
      text: 'totals are 5'
    })
  })

  it('collapses the whitespace a long message carries', () => {
    expect(quoteFromMessage(message('line one\n\nline   two'), 'Builder', 1).text).toBe('line one line two')
  })

  it('cuts a long line to the log budget', () => {
    const quoted = quoteFromMessage(message('x'.repeat(400)), 'Builder', 1).text

    expect(quoted).toHaveLength(GROUP_QUOTE_TEXT_CHARS)
    expect(quoted.endsWith('…')).toBe(true)
  })

  it('leaves a line exactly at the budget alone', () => {
    const exact = 'y'.repeat(GROUP_QUOTE_TEXT_CHARS)

    expect(quoteFromMessage(message(exact), 'Builder', 1).text).toBe(exact)
  })
})
