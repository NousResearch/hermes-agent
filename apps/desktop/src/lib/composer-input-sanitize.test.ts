import { describe, expect, it } from 'vitest'

import {
  collapseRepeatedInputArtifacts,
  sanitizeComposerInput,
  stripLeakedBracketedPasteWrappers
} from './composer-input-sanitize'

describe('stripLeakedBracketedPasteWrappers', () => {
  it('leaves plain text unchanged', () => {
    expect(stripLeakedBracketedPasteWrappers('hello world')).toBe('hello world')
  })

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
  it('normalizes wrappers and repeated artifact tails together', () => {
    const corrupted = 'hello[' + '~[[e'.repeat(8)
    expect(sanitizeComposerInput(corrupted)).toBe('hello')
  })
})
