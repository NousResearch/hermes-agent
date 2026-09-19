import React from 'react'
import { describe, expect, it } from 'vitest'

import { voteDirection, voteDirectionFromNodes } from './bidi'

describe('voteDirection', () => {
  it('resolves pure English to ltr', () => {
    expect(voteDirection('Task Manager')).toBe('ltr')
    expect(voteDirection('End task to see details')).toBe('ltr')
  })

  it('resolves pure Persian to rtl', () => {
    expect(voteDirection('سلام به همه')).toBe('rtl')
    expect(voteDirection('هیچ‌کدام نماند')).toBe('rtl')
  })

  it('every Persian-specific glyph votes rtl (transport integrity probe)', () => {
    for (const ch of ['ر', 'پ', 'ژ', 'گ', 'چ', 'ی', 'ک']) {
      expect(voteDirection(ch)).toBe('rtl')
    }
  })

  it('resolves neutral-only text to ltr', () => {
    expect(voteDirection('')).toBe('ltr')
    expect(voteDirection('123 …?!')).toBe('ltr')
    // Arabic-block punctuation never votes.
    expect(voteDirection('؟!')).toBe('ltr')
  })

  it('lets the majority win regardless of the first word', () => {
    // Starts with English (11 LTR letters) but Persian-majority (16 RTL).
    expect(voteDirection('Task Manager را باز کن و ادامه بده')).toBe('rtl')
    // Starts with Persian (7 RTL) but English-majority (28 LTR).
    expect(voteDirection('را باز کن End task to see the details panel')).toBe('ltr')
  })

  it('breaks ties toward rtl', () => {
    // 2 RTL letters (آب) vs 2 LTR letters (go).
    expect(voteDirection('آب go')).toBe('rtl')
  })

  it('resolves the reported sentence to rtl', () => {
    expect(
      voteDirection('Task Manager را باز کن، در تب Details همه Hermes.exe ها را End task کن تا هیچ‌کدام نماند.')
    ).toBe('rtl')
  })
})

describe('voteDirectionFromNodes', () => {
  it('votes over nested children', () => {
    expect(voteDirectionFromNodes(React.createElement('b', null, 'سلام دنیا'))).toBe('rtl')
    expect(voteDirectionFromNodes(['hello ', React.createElement('b', null, 'world')])).toBe('ltr')
  })

  it('excludes code spans from the vote', () => {
    // Without the exclusion the 10 Latin letters of the command outvote
    // the 8 Persian ones and flip the paragraph to ltr.
    const children = [
      React.createElement('code', { key: 'c' }, 'npm install'),
      ' را اجرا کن'
    ]

    expect(voteDirectionFromNodes(children)).toBe('rtl')
  })

  it('excludes reference chips from the vote', () => {
    const children = [
      'ببین ',
      React.createElement('span', { 'data-ref-text': '@file:`x.ts`', key: 'r' }, 'x.ts'),
      ' لطفا'
    ]

    expect(voteDirectionFromNodes(children)).toBe('rtl')
  })
})
