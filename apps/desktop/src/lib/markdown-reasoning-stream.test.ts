import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

/**
 * Reasoning blocks on the STREAMING surface.
 *
 * Models that inline their chain of thought in the answer channel (`<thinking>…`
 * — DeepSeek/R1-style, and anything relaying a reasoning model onto a chat wire)
 * stream the open tag first and the close tag much later, if at all. Two things
 * follow from `preprocessMarkdown` running on the accumulated text on every
 * flush, and both were visible in the chat:
 *
 *   1. The chain of thought rendered as user-visible prose until the close tag
 *      landed, and when it did the whole span was deleted in one frame — taking
 *      text the reader had already read with it (the reported mid-reply
 *      "truncation" on long answers).
 *   2. The strip ate the whitespace on its seam, so a block sitting BETWEEN two
 *      words fused them: `no` + `Hermes` rendered as `noHermes`.
 */
const normalize = (value: string) => value.replace(/\s+/g, ' ').trim()

const REASONING = 'let me think about the render pipeline in detail'

describe('reasoning blocks in streamed markdown', () => {
  it('does not fuse the words a stripped block separated', () => {
    expect(preprocessMarkdown('no<thinking>hmm</thinking> Hermes Desktop')).toBe('no Hermes Desktop')
    expect(preprocessMarkdown('componentes<analysis>e</analysis>-chave aqui')).toBe('componentes -chave aqui')
    expect(preprocessMarkdown('backend e o <thinking>x</thinking>useSmoothReveal')).toBe('backend e o useSmoothReveal')
  })

  it('hides an unterminated reasoning block instead of rendering it', () => {
    expect(preprocessMarkdown('<thinking>let me think about the render pipeline')).toBe('')
    expect(preprocessMarkdown('<thinking>')).toBe('')
    expect(preprocessMarkdown('Resposta final.\n<thinking>checking the render pipeline')).toBe('Resposta final.\n')
    expect(preprocessMarkdown('Resposta final.\n<thinking>')).toBe('Resposta final.\n')
  })

  it('leaves prose that merely mentions a tag alone', () => {
    const inline = 'The model writes <thinking> tags inline in its output, and closes them too.'
    const quoted = 'O texto acima explica o formato do bloco <thinking> sem nunca fechá-lo'

    expect(preprocessMarkdown(inline)).toBe(inline)
    expect(preprocessMarkdown(quoted)).toBe(quoted)
  })

  it('drops the block for a closed pair, however it is split', () => {
    expect(preprocessMarkdown(`<thinking>${REASONING}</thinking>\n\nResposta final com acentuação.`)).toBe(
      'Resposta final com acentuação.'
    )
    expect(preprocessMarkdown(`Resposta: <analysis>${REASONING}</analysis>e o resto.`)).toBe('Resposta: e o resto.')
  })

  it('never shows the chain of thought at any delta boundary', () => {
    const stream = `<thinking>${REASONING}</thinking>Resposta final com acentuação, referência e ação.`
    const answers: string[] = []
    let accumulated = ''

    for (let index = 0; index < stream.length; index += 3) {
      accumulated = stream.slice(0, index + 3)
      const visible = preprocessMarkdown(accumulated)

      // The reasoning text and the completed tag must never reach the surface,
      // at any flush — not only once the block has closed. (A half-arrived tag
      // is not a tag yet; it renders as nothing, and is covered in the DOM test.)
      expect(visible).not.toContain('let me think')
      expect(visible).not.toContain('render pipeline')
      expect(visible).not.toMatch(/<(\/?)(think|thinking|reasoning|scratchpad|analysis)>/i)

      // And nothing that WAS visible may disappear as the stream grows: every
      // visible fragment is a prefix of what the next frame shows. A half-arrived
      // tag (`<thin`, no `>` yet) is not a tag this pass can see — the DOM renders
      // those 1-4 characters as nothing the moment the tag completes, which the
      // surface test pins end-to-end — so those frames are not part of the curve.
      if (/<[a-z]*$/i.test(accumulated)) {
        continue
      }

      const next = normalize(visible)
      const previous = answers.at(-1)

      if (previous !== undefined) {
        expect(next.startsWith(previous)).toBe(true)
      }

      answers.push(next)
    }

    expect(normalize(preprocessMarkdown(accumulated))).toBe('Resposta final com acentuação, referência e ação.')
  })
})
