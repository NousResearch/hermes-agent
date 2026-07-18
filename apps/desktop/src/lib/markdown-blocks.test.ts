import { parseMarkdownIntoBlocks } from '@assistant-ui/react-streamdown'
import { describe, expect, it } from 'vitest'

import { parseMarkdownIntoBlocksCached } from './markdown-blocks'

// The contract: streaming through the cached splitter (one call per growing
// prefix, exactly how Streamdown calls it per flush) must produce, at every
// step, the same blocks as a fresh full lex of that prefix. Byte equality —
// a divergence would change what the memoized block renderer paints.

const CORPUS = `# Heading

Intro paragraph with **bold**, [a link](https://example.com), \`inline\` and $x^2$ math.

- list item one
- list item two
  - nested item

1. ordered a

2. loose ordered b

\`\`\`python
def f(x):
    return x * 2  # comment with \`\`\` inside string? no — fence chars below
\`\`\`

A paragraph that will be followed by a setext underline
===

| col a | col b |
|---|---|
| 1 | 2 |
| 3 | 4 |

> blockquote line one
> blockquote line two
with a lazy continuation line

<div class="raw">
html block content
</div>

$$
\\int_0^1 x\\,dx = \\tfrac12
$$

Final paragraph after everything, long enough to stream in pieces so the tail
block keeps getting reinterpreted while earlier blocks stay settled.
`

// Deterministic PRNG so failures reproduce.
function mulberry32(seed: number) {
  let a = seed

  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Push the text past the cache MIN_LENGTH thresholds so the incremental
// path actually engages.
const LONG_CORPUS = Array.from({ length: 6 }, () => CORPUS).join('\n')

describe('parseMarkdownIntoBlocksCached', () => {
  it('matches a full lex at every random streaming cut (property)', () => {
    for (let seed = 1; seed <= 5; seed++) {
      const rand = mulberry32(seed)
      let cursor = 0

      while (cursor < LONG_CORPUS.length) {
        cursor = Math.min(LONG_CORPUS.length, cursor + 1 + Math.floor(rand() * 120))
        const prefix = LONG_CORPUS.slice(0, cursor)

        expect(parseMarkdownIntoBlocksCached(prefix)).toEqual(parseMarkdownIntoBlocks(prefix))
      }
    }
  })

  it('matches a full lex when streaming token-by-token through a fence boundary', () => {
    const base = `${'settled paragraph one.\n\n'.repeat(100)}opening a fence now:\n`
    const tail = '```js\nconst a = 1\nconst b = 2\n```\n\nafter the fence\n'

    for (let i = 1; i <= tail.length; i++) {
      const text = base + tail.slice(0, i)

      expect(parseMarkdownIntoBlocksCached(text)).toEqual(parseMarkdownIntoBlocks(text))
    }
  })

  it('reconstructs the input exactly (join property the offsets rely on)', () => {
    const blocks = parseMarkdownIntoBlocksCached(LONG_CORPUS)

    expect(blocks.join('')).toBe(LONG_CORPUS)
  })

  it('falls back to a full lex for non-append rewrites (edit / branch swap)', () => {
    const grown = `${LONG_CORPUS}\n\nappended tail paragraph`
    parseMarkdownIntoBlocksCached(grown)

    // A REWRITE that shares no prefix lineage must still be correct.
    const rewritten = `completely different start\n\n${LONG_CORPUS.slice(500)}`

    expect(parseMarkdownIntoBlocksCached(rewritten)).toEqual(parseMarkdownIntoBlocks(rewritten))
  })
})
