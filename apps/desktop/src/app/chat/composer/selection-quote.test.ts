import { describe, expect, it } from 'vitest'

import { selectionQuote } from './selection-quote'

describe('selectionQuote', () => {
  it('keeps a non-editable transcript selection and gives it a stable label', () => {
    expect(selectionQuote('  explain this reply  ', 'Chat transcript', false)).toEqual({
      label: 'Chat transcript',
      text: 'explain this reply'
    })
  })

  it('does not quote selections from an editable control', () => {
    expect(selectionQuote('draft text', 'notes.ts', true)).toBeNull()
  })

  it('falls back to a stable preview label', () => {
    expect(selectionQuote('selected preview', '', false)).toEqual({ label: 'Preview', text: 'selected preview' })
  })
})
