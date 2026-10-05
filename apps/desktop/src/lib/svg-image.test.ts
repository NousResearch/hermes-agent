import { describe, expect, it } from 'vitest'

import { normalizeSvgSize, selfCloseVoidTags, svgSize } from './svg-image'

// Real mermaid 11.16 render output shape (verified against the installed
// package): width="100%" + inline style="max-width: Npx" + viewBox.
const MERMAID_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="100%" class="flowchart" style="max-width: 260.34375px;" viewBox="0 0 260.34375 70" role="graphics-document document"><g><rect x="0" y="0" width="260.34375" height="70" fill="#eee"/></g></svg>`

// Mermaid serialises a label's <br/> as a bare HTML <br> inside a
// foreignObject label — the shape that breaks XML parsing (#133089).
const MERMAID_BR_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 260 70"><foreignObject><div xmlns="http://www.w3.org/1999/xhtml" class="label"><p><span class="nodeLabel">ice point<br>limit-up</span></p></div></foreignObject></svg>`

describe('svgSize', () => {
  it('reads explicit pixel width/height', () => {
    expect(svgSize('<svg width="312" height="90"><rect/></svg>')).toEqual({ width: 312, height: 90 })
  })

  it('falls back to the viewBox when width is a percentage (mermaid)', () => {
    expect(svgSize(MERMAID_SVG)).toEqual({ width: 260.34375, height: 70 })
  })

  it('falls back to the viewBox when width/height are absent', () => {
    expect(svgSize('<svg viewBox="0 0 400 100"><rect/></svg>')).toEqual({ width: 400, height: 100 })
  })

  it('falls back to a default size when nothing usable exists', () => {
    expect(svgSize('<svg><rect/></svg>')).toEqual({ width: 800, height: 600 })
  })

  it('treats mixed-unit widths as absent (not parseFloat(100) == 100)', () => {
    expect(svgSize('<svg width="100%" height="70" viewBox="0 0 500 200"><rect/></svg>')).toEqual({
      width: 500,
      height: 200
    })
  })
})

describe('normalizeSvgSize', () => {
  it('replaces a 100% width with the viewBox pixel size', () => {
    const out = normalizeSvgSize(MERMAID_SVG)

    expect(out).toContain('width="260.34375"')
    expect(out).toContain('height="70"')
    expect(out).not.toContain('width="100%"')
    expect(out).toContain('viewBox="0 0 260.34375 70"')
    expect(out).toContain('role="graphics-document document"')
  })

  it('leaves an explicit pixel height when only width is a percentage', () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="70" viewBox="0 0 500 200"><rect/></svg>'

    const out = normalizeSvgSize(svg)

    expect(out).toContain('width="500"')
    expect(out).toContain('height="70"')
    expect(out).not.toContain('height="200"')
  })

  it('leaves svgs without a percentage width untouched', () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="312" viewBox="0 0 312 90"><rect/></svg>'

    expect(normalizeSvgSize(svg)).toBe(svg)
  })

  it('leaves svgs with a percentage width but no viewBox untouched', () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100%"><rect/></svg>'

    expect(normalizeSvgSize(svg)).toBe(svg)
  })
})

describe('selfCloseVoidTags', () => {
  it('self-closes bare <br> tags so the svg parses as XML (#133089)', () => {
    const out = selfCloseVoidTags(MERMAID_BR_SVG)

    expect(out).toContain('ice point<br/>limit-up')

    const root = new DOMParser().parseFromString(out, 'image/svg+xml').documentElement

    expect(root.tagName).toBe('svg')
  })

  it('leaves already-closed void tags byte-identical', () => {
    const svg = '<svg><text>ice point<br/>limit-up</text><img src="x.png"/><hr /></svg>'

    expect(selfCloseVoidTags(svg)).toBe(svg)
  })

  it('self-closes a void tag that carries attributes', () => {
    const out = selfCloseVoidTags('<svg><br style="font-weight: bold"></svg>')

    expect(out).toBe('<svg><br style="font-weight: bold"/></svg>')
  })

  it('does not touch longer tags that start with the same letters', () => {
    const svg = '<svg><break>not a void tag</break><inputx/></svg>'

    expect(selfCloseVoidTags(svg)).toBe(svg)
  })

  it('runs before normalizeSvgSize so the size pass sees a parsable svg', () => {
    const out = normalizeSvgSize(selfCloseVoidTags(MERMAID_BR_SVG))

    expect(out).toContain('width="260"')
    expect(out).toContain('height="70"')
    expect(out).toMatch(/ice point<br\s*\/>limit-up/)
  })
})
