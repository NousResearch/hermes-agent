import { describe, expect, it } from 'vitest'

import {
  collapseRepeatedInputArtifacts,
  sanitizeComposerInput,
  stripLeakedBracketedPasteWrappers
} from './composer-input-sanitize'

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
  it('preserves every prefix of a clock time range', () => {
    for (const text of ['21:00~21:02', '21:01~  21:02', '9:00~9:30']) {
      for (let end = 1; end <= text.length; end++) {
        const prefix = text.slice(0, end)
        expect(sanitizeComposerInput(prefix)).toBe(prefix)
      }
    }
  })

  it('distinguishes clock text from paste wrappers', () => {
    expect(sanitizeComposerInput('note:00~hello01~')).toBe('note:hello')
    expect(sanitizeComposerInput('00~21:00~21:02 and 00~x01~')).toBe('21:00~21:02 and x')

    for (const text of ['id01~21:00~21:02', '21:00~21:02[200~tag']) {
      expect(sanitizeComposerInput(text)).toBe(text)
    }
  })

  it('normalizes wrappers and repeated artifact tails together', () => {
    const corrupted = 'hello[' + '~[[e'.repeat(8)
    expect(sanitizeComposerInput(corrupted)).toBe('hello')
  })
})
