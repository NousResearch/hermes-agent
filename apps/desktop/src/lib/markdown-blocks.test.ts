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

  it('matches a full lex when a trailing setext underline merges the previous block (regression)', () => {
    // A trailing `-`/`=` line is a setext underline of the block ABOVE it, so
    // appending to it can retroactively merge the previous parse's LAST TWO
    // blocks into one. Cached `"…#e\n5\n-"` lexes to [ …, "#e\n", "5\n-" ], but
    // grown to `"…#e\n5\n-p2=kj:c"` collapses `#e`/`5\n-` into one block. The
    // old boundary dropped only the single last content block, so it reused a
    // `"#e\n"` block that no longer exists. `blocks.join('') === text` still
    // holds for the wrong split, so the reconstruction guard cannot catch it.
    // The settled prefix pushes the text past the append-cache threshold so
    // the incremental path actually engages.
    const settled = 'settled line paragraph text.\n\n'.repeat(80)
    const prev = `${settled}#e\n5\n-`
    const grown = `${prev}p2=kj:c`

    // Seed the append cache with `prev`, then grow it — the exact two-call
    // sequence a streaming flush produces.
    parseMarkdownIntoBlocksCached(prev)

    expect(parseMarkdownIntoBlocksCached(grown)).toEqual(parseMarkdownIntoBlocks(grown))
  })

  // 12 seeds × 500 growing prefixes is ~6000 full+cached lexes; it first trips
  // the pre-fix boundary at seed 11 / step 257, so the workload can't shrink
  // without gutting the guard. The work is bounded but exceeds one test's 5s
  // default budget, so raise the timeout rather than weaken the coverage.
  it('matches a full lex at every char-level streaming cut over noisy markdown (property fuzz)', () => {
    // Character-level append fuzz over the markdown control alphabet — the
    // harness that surfaced the setext-underline merge above. Growing a single
    // lineage one small chunk at a time keeps `startsWith` lineage intact so
    // the incremental path runs on nearly every step; each prefix must
    // deep-equal a fresh full lex.
    const alphabet = '\n `#*-_>[]()|~:=abcdefghijklmnopqrstuvwxyz0123456789'

    for (let seed = 1; seed <= 12; seed++) {
      const rand = mulberry32(seed)
      // Seed past the append-cache threshold so the incremental path engages.
      let text = `seed ${seed}\n\n`.repeat(180)

      for (let step = 0; step < 500; step++) {
        const n = 1 + Math.floor(rand() * 24)
        let chunk = ''

        for (let j = 0; j < n; j++) {
          chunk += alphabet[Math.floor(rand() * alphabet.length)]
        }

        text += chunk

        expect(parseMarkdownIntoBlocksCached(text)).toEqual(parseMarkdownIntoBlocks(text))
      }
    }
  }, 30_000)
})
