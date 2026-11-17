import { describe, expect, it } from 'vitest'

import {
  collapseRepeatedInputArtifacts,
  sanitizeComposerInput,
  stripLeakedBracketedPasteWrappers
} from './composer-input-sanitize'

// Range corpus from Songkey1 in leve-team/hermes-agent#12; mirrored in Python.
const ORDINARY_RANGE_TEXTS = [
  '회의는 09:00~18:00 입니다',
  '오전 10:00~11:00, 오후 2:00~3:00',
  '13:00~ 시작',
  '08:01~ 출발',
  '2001~ 2005',
  '10~20개',
  '(100~200)',
  '1.00~1.01',
  'tcp:8000~8010',
  ':00~:30 매시\n:00~ 정각',
  'v1.01~beta',
  '00~06시 심야, 1월 01~ 05일',
  '기간: 2024-01~',
  '값은 [200~300] 사이, [201~ 300]',
  '営業時間は9:00~18:00です',
  'Open 9:00~17:00, pages 100~200'
]

describe('stripLeakedBracketedPasteWrappers', () => {
  it('strips canonical escape wrappers', () => {
    expect(stripLeakedBracketedPasteWrappers('\x1b[200~hello\x1b[201~')).toBe('hello')
  })

  it('keeps embedded literal bracket forms', () => {
    const text = 'literal[200~tag and literal[201~tag should stay'
    expect(stripLeakedBracketedPasteWrappers(text)).toBe(text)
  })
})

describe('collapseRepeatedInputArtifacts', () => {
  it('removes the desktop corruption tail from #62557', () => {
    const prefix = '需要时随时叫我。'
    const tail = '[e~[[e' + '~[[e'.repeat(20)
    expect(collapseRepeatedInputArtifacts(prefix + tail)).toBe(prefix)
  })

  it('preserves a mid-string marker followed by valid suffix', () => {
    const text = 'notes ~[[e more text here'
    expect(collapseRepeatedInputArtifacts(text)).toBe(text)
  })

  it('preserves trailing punctuation that is not the corruption signature', () => {
    expect(collapseRepeatedInputArtifacts('wait....')).toBe('wait....')
  })

  it('does not strip when fewer than minRepeats markers appear at the tail', () => {
    const text = 'hello~[[e~[[e'
    expect(collapseRepeatedInputArtifacts(text)).toBe(text)
  })
})

describe('sanitizeComposerInput', () => {
  it.each(ORDINARY_RANGE_TEXTS)('preserves ordinary range text: %s', text => {
    expect(sanitizeComposerInput(text)).toBe(text)

    for (const [start, end] of [
      ['\x1b[200~', '\x1b[201~'],
      ['^[[200~', '^[[201~']
    ]) {
      expect(sanitizeComposerInput(start + text + end)).toBe(text)
    }
  })

  it('preserves every prefix of a time or date range', () => {
    for (const text of ['21:00~21:02', '21:01~  21:02', '9:00~9:30', '12.01~ 12.15', '2024-01~2024-02']) {
      for (let end = 1; end <= text.length; end++) {
        const prefix = text.slice(0, end)
        expect(sanitizeComposerInput(prefix)).toBe(prefix)
      }
    }
  })

  it('preserves dotted clock time during incremental input', () => {
    for (const text of ['12.01~', '12.01~ 12.15']) {
      expect(sanitizeComposerInput(text)).toBe(text)
    }
  })

  it('distinguishes numeric text from paste wrappers', () => {
    expect(sanitizeComposerInput('note:00~hello01~')).toBe('note:hello')
    expect(sanitizeComposerInput('00~21:00~21:02 and 00~x01~')).toBe('00~21:00~21:02 and x')

    for (const text of ['id01~21:00~21:02', '21:00~21:02[200~tag', '00~5 items', 'count 501~']) {
      expect(sanitizeComposerInput(text)).toBe(text)
    }
  })

  it.each([
    ['\u001b[200~hello\u001b[201~', 'hello'],
    ['^[[200~hello^[[201~', 'hello'],
    ['prefix [200~hello[201~ suffix', 'prefix hello suffix'],
    ['prefix 00~hello world01~ suffix', 'prefix hello world suffix'],
    ['^[[200~line 1\nline 2\nline 3^[[201~', 'line 1\nline 2\nline 3'],
    ['[200~line 1\nline 2\nline 3[201~', 'line 1\nline 2\nline 3']
  ])('cleans unambiguous paste wrappers: %s', (text, expected) => {
    expect(sanitizeComposerInput(text)).toBe(expected)
  })

  it('normalizes wrappers and repeated artifact tails together', () => {
    const corrupted = 'hello[' + '~[[e'.repeat(8)
    expect(sanitizeComposerInput(corrupted)).toBe('hello')
  })
})
