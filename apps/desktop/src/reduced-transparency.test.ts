// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const STYLES = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'styles.css'), 'utf8').replace(
  /\/\*[\s\S]*?\*\//g,
  ''
)

const REDUCED_TRANSPARENCY_MEDIA = /@media\s*\(\s*prefers-reduced-transparency\s*:\s*reduce\s*\)\s*\{/g

/** The text between the brace at `open` and its matching close. */
function blockBody(css: string, open: number): string {
  let depth = 0

  for (let index = open; index < css.length; index += 1) {
    if (css[index] === '{') {
      depth += 1
    } else if (css[index] === '}' && (depth -= 1) === 0) {
      return css.slice(open + 1, index)
    }
  }

  throw new Error('Unbalanced block in styles.css')
}

describe('reduced-transparency gate', () => {
  // Static mask-image fades build CoreAnimation filter layers (ImagingNode) that
  // WindowServer re-renders every display refresh. The blanket gate must drop
  // them alongside backdrop-filter, or the compositor keeps paying for the
  // nested chain on a static scene.
  it('drops backdrop-filter and mask-image on the OS reduce-transparency signal', () => {
    const blocks = [...STYLES.matchAll(REDUCED_TRANSPARENCY_MEDIA)].map(match =>
      blockBody(STYLES, match.index + match[0].length - 1)
    )

    expect(blocks.length).toBe(1)
    const body = blocks[0]

    expect(body).toContain('backdrop-filter: none')
    expect(body).toContain('-webkit-backdrop-filter: none')
    expect(body).toContain('mask-image: none')
    expect(body).toContain('-webkit-mask-image: none')
  })
})
