import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

// preprocessMarkdown runs on the accumulated text every streaming flush, so the
// reasoning strip has to behave on a text whose close tag has not arrived yet.
describe('reasoning blocks in streamed markdown', () => {
  it('does not fuse the words a stripped block separated', () => {
    expect(preprocessMarkdown('no<thinking>hmm</thinking> Hermes Desktop')).toBe('no Hermes Desktop')
    expect(preprocessMarkdown('backend e o <thinking>x</thinking>useSmoothReveal')).toBe('backend e o useSmoothReveal')
    expect(preprocessMarkdown('<thought>x</thought>\n\nResposta final.')).toBe('Resposta final.')
    expect(preprocessMarkdown('no<thinking>a</thinking><think>b</think>Hermes')).toBe('no Hermes')
  })

  it('hides an unterminated block at a block boundary but keeps a mid-sentence mention', () => {
    expect(preprocessMarkdown('<thinking>let me think about the render pipeline')).toBe('')
    expect(preprocessMarkdown('Resposta final.\n<reasoning_scratchpad>checking')).toBe('Resposta final.\n')
    expect(preprocessMarkdown('Answer.\n<thin')).toBe('Answer.\n')
    expect(preprocessMarkdown('Answer.\n<div')).toBe('Answer.\n<div')

    const quoted = 'O texto acima explica o formato do bloco <thinking> sem nunca fechá-lo'
    // The mid-sentence mention is escaped to entities by the #53953 unknown-tag
    // pass — the renderer would otherwise read it as an unclosed tag — so the
    // round-trip differs from the source by exactly that escape, and the
    // escaped form paints identically as literal text.
    expect(preprocessMarkdown(quoted)).toBe(
      'O texto acima explica o formato do bloco &lt;thinking&gt; sem nunca fechá-lo'
    )
  })
})
