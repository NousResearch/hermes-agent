import { describe, expect, it } from 'vitest'

import { normalizeSvgSize } from './svg-image'

// Real mermaid 11.16 output shape for an oversized diagram: width="100%" plus
// an inline max-width that pins the intrinsic width in pixels.
const WIDE_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" width="100%" class="flowchart" style="max-width: 4758.394px;" viewBox="0 0 4758.394 1671.458" role="graphics-document document"><g><rect width="4758" height="1671"/></g></svg>'

describe('normalizeSvgSize', () => {
  it('neutralises an inline pixel max-width so CSS width caps can apply', () => {
    const out = normalizeSvgSize(WIDE_SVG)

    // The inline max-width used to beat the max-w-full / max-w-[85vw] caps on
    // both the inline preview and the overlay, so a wide diagram rendered at
    // intrinsic width.
    expect(out).toContain('max-width: 100%')
    expect(out).not.toContain('max-width: 4758')
    expect(out).toContain('width="4758.394"')
    expect(out).toContain('height="1671.458"')
  })

  it('keeps an inline max-width that is already relative', () => {
    const svg =
      '<svg xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width: 100%;" viewBox="0 0 300 100"><g/></svg>'

    const out = normalizeSvgSize(svg)

    expect(out).toContain('max-width: 100%')
  })
})
