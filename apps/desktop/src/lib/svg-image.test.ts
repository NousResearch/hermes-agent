import { describe, expect, it } from 'vitest'

import { normalizeSvgSize, svgSize } from './svg-image'

// Real mermaid 11.16 render output shape (verified against the installed
// package): width="100%" + inline style="max-width: Npx" + viewBox.
const MERMAID_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="100%" class="flowchart" style="max-width: 260.34375px;" viewBox="0 0 260.34375 70" role="graphics-document document"><g><rect x="0" y="0" width="260.34375" height="70" fill="#eee"/></g></svg>`

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
    const svg =
      '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="70" viewBox="0 0 500 200"><rect/></svg>'
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
