import { describe, expect, it } from 'vitest'

import { normalizeSvgSize, svgSize, xmlWellFormedSvg } from './svg-image'

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

// Real mermaid 11.16 label island for `A["a<br/>b&nbsp;c"]` (verified against
// the installed package): the HTML serialisation leaves `<br>` open and writes
// the non-breaking space as `&nbsp;`, and neither is well-formed XML. Every
// strict `image/svg+xml` parse degrades on it — and Blink fails the same way
// when the string is handed to an <img> as a data: URI, which is how a diagram
// silently stops rendering (broken image, no error anywhere).
const MERMAID_LABEL_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 260.34375 70"><g class="label"><foreignObject width="19.109375" height="48"><div xmlns="http://www.w3.org/1999/xhtml" style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;"><span class="nodeLabel"><p>a<br>b&nbsp;c</p></span></div></foreignObject></g></svg>`

describe('xmlWellFormedSvg', () => {
  const parseXml = (svg: string) => new DOMParser().parseFromString(svg, 'image/svg+xml')

  it('makes a mermaid label with <br> and &nbsp; parse as an <svg> instead of a parsererror page', () => {
    expect(parseXml(MERMAID_LABEL_SVG).documentElement.tagName).not.toBe('svg')
    expect(parseXml(xmlWellFormedSvg(MERMAID_LABEL_SVG)).documentElement.tagName).toBe('svg')
    expect(normalizeSvgSize(xmlWellFormedSvg(MERMAID_LABEL_SVG))).toContain('width="260.34375"')
  })

  it('keeps the label content: the line break and the non-breaking space', () => {
    const label = parseXml(xmlWellFormedSvg(MERMAID_LABEL_SVG)).querySelector('p')

    expect(label?.querySelectorAll('br')).toHaveLength(1)
    expect(label?.textContent).toBe('ab c')
  })
})
