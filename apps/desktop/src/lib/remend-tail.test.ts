import { parseMarkdownIntoBlocks } from '@assistant-ui/react-streamdown'
import remend from 'remend'
import { describe, expect, it } from 'vitest'

import { findRemendWindowStart, tailBoundedRemend } from './remend-tail'

const CORPUS = `# Heading one

Intro paragraph with **bold**, *italic*, \`inline code\`, and a [link](https://example.com).

## Code

\`\`\`python
def main():
    cost = "$5"
    print(f"total: $\{cost}")
\`\`\`

Some text after the fence with $x^2 + y^2$ inline math.

$$
\\int_0^1 f(x) dx
$$

- list item one with **bold**
- list item two

| col a | col b |
| ----- | ----- |
| 1     | 2     |

~~~js
const s = \`template \${value}\`
~~~

Final paragraph with ~~strike~~ and unfinished [link text](https://exa
`

/**
 * Render-equivalence oracle: full-text remend and tail-bounded remend may
 * differ in raw string output ONLY in ways that cannot affect rendering —
 * i.e. after block splitting, every block must be identical. (Streamdown
 * renders blocks independently, so block-level equality IS render equality.)
 */
function blocksOf(text: string): string[] {
  return parseMarkdownIntoBlocks(text)
}

describe('tailBoundedRemend', () => {
  it('matches full remend block output at every streaming prefix', () => {
    for (let end = 1; end <= CORPUS.length; end++) {
      const prefix = CORPUS.slice(0, end)
      const full = blocksOf(remend(prefix))
      const tail = blocksOf(tailBoundedRemend(prefix))

      expect(tail, `prefix length ${end}: ${JSON.stringify(prefix.slice(-60))}`).toEqual(full)
    }
  })

  it('repairs an unclosed fence opened early in a long message', () => {
    const text = `intro\n\n\`\`\`python\n${'x = 1\n'.repeat(500)}print("$dollar")`
    const repaired = tailBoundedRemend(text)

    expect(blocksOf(repaired)).toEqual(blocksOf(remend(text)))
    // the window must reach back to the fence opener
    expect(findRemendWindowStart(text)).toBe(text.indexOf('```python'))
  })

  it('bounds the window to the tail paragraph when no fence is open', () => {
    const text = `para one\n\npara two\n\npara three with **bold`
    const start = findRemendWindowStart(text)

    expect(start).toBe(text.indexOf('para three'))
    expect(tailBoundedRemend(text)).toBe(remend(text))
  })

  it('widens the window across an open $$ math block', () => {
    const text = `before\n\n$$\n\\frac{a}{b}`
    const start = findRemendWindowStart(text)

    expect(start).toBeLessThanOrEqual(text.indexOf('$$'))
    expect(blocksOf(tailBoundedRemend(text))).toEqual(blocksOf(remend(text)))
  })

  it('handles closed constructs without modification', () => {
    const text = `done **bold** and \`code\`\n\n\`\`\`js\nconst a = 1\n\`\`\`\n\nlast line.`

    expect(tailBoundedRemend(text)).toBe(text)
  })

  it('intentionally diverges from full remend on cross-block dangling openers', () => {
    // Full remend scans the whole document and appends `**` for an opener
    // left dangling in an EARLIER block, dumping stray asterisks into the
    // unrelated tail block ("|**"). Because Streamdown splits into blocks
    // after the repair, that opener never renders as bold either way — the
    // tail-bounded result is the cleaner of the two. This test documents
    // the divergence so a future remend upgrade that changes the behavior
    // gets noticed.
    const text = `- item with **dangling\n- item two\n\n|`

    expect(remend(text).endsWith('|**')).toBe(true)
    expect(tailBoundedRemend(text).endsWith('|')).toBe(true)
    expect(tailBoundedRemend(text).endsWith('|**')).toBe(false)
  })
})
