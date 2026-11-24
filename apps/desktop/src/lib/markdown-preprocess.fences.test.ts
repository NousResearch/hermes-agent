import { describe, expect, it } from 'vitest'

import { normalizeFilePreviewMath, preprocessMarkdown } from '@/lib/markdown-preprocess'

// Listings whose own text carries fence characters: a ``` mid-line, and the
// inner fences of a markdown example fenced with a longer or other marker.
// Each must reach the renderer byte-for-byte, with the prose after it outside.
const LISTINGS = [
  ['```python', 'def is_fence(line):', '    return line.startswith("```")', 'first = parts[0]', 'cost = "$5"', '```'],
  ['````markdown', '```js', 'const a = arr[0];', '```', '````'],
  ['~~~md', '```sh', 'export A=$5; echo \\(x\\)', '```', '~~~'],
  ['```js', 'const tilde = "~~~ and $5 [1]"', '```']
].map(lines => lines.join('\n'))

describe('fenced code segmentation', () => {
  it('keeps a listing that contains fence characters intact in chat markdown', () => {
    for (const listing of LISTINGS) {
      const output = preprocessMarkdown(`Example:\n\n${listing}\n\nAfter the example.`)

      expect(output).toContain(listing)
      expect(output.endsWith(`${listing}\n\nAfter the example.`)).toBe(true)
    }
  })

  it('keeps the same listings intact in a file preview, CRLF included', () => {
    for (const listing of [...LISTINGS, '```sh\r\nexport A=$5\r\n```']) {
      const input = `Intro \\(y\\)\n\n${listing}\n\nOutro`

      expect(normalizeFilePreviewMath(input)).toBe(`Intro $y$\n\n${listing}\n\nOutro`)
    }
  })
})
