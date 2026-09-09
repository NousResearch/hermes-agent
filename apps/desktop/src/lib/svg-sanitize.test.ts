import { describe, expect, it } from 'vitest'

import { sanitizeSvgMarkup } from './svg-sanitize'

function sanitizedSvg(markup: string): SVGSVGElement {
  const container = document.createElement('div')

  container.innerHTML = sanitizeSvgMarkup(markup)

  const svg = container.querySelector('svg')

  if (!svg) {
    throw new Error('expected sanitized SVG root')
  }

  return svg
}

describe('sanitizeSvgMarkup', () => {
  it('retains safe shapes, text, and same-document gradients', () => {
    const svg = sanitizedSvg(`<svg viewBox="0 0 20 20">
  <defs><linearGradient id="g"><stop offset="0" stop-color="#fff"/></linearGradient></defs>
  <rect id="shape" width="20" height="20" fill="url(#g)"/>
  <text id="label">safe</text>
</svg>`)

    expect(svg.querySelector('linearGradient#g')).not.toBeNull()
    expect(svg.querySelector('#shape')?.getAttribute('fill')).toBe('url(#g)')
    expect(svg.querySelector('#label')?.textContent).toBe('safe')
  })

  it('keeps local geometry references but intentionally removes filter graphs and references', () => {
    const svg = sanitizedSvg(`<svg>
  <defs>
    <clipPath id="clip"><circle r="1"/></clipPath>
    <filter id="shadow"><feGaussianBlur stdDeviation="1"/></filter>
  </defs>
  <rect id="shape" clip-path="url(#clip)" filter="url(#shadow)"/>
</svg>`)

    expect(svg.querySelector('clipPath#clip')).not.toBeNull()
    expect(svg.querySelector('#shape')?.getAttribute('clip-path')).toBe('url(#clip)')
    expect(svg.querySelector('filter')).toBeNull()
    expect(svg.querySelector('#shape')?.hasAttribute('filter')).toBe(false)
  })

  it.each([
    ['https image', '<image href="https://example.com/a.svg"/>'],
    ['http image', '<image href="http://example.com/a.svg"/>'],
    ['http localhost image', '<image href="http://localhost:8080/private"/>'],
    ['file image', '<image href="file:///etc/passwd"/>'],
    ['data image', '<image href="data:image/svg+xml,boom"/>'],
    ['protocol-relative use', '<use href="//example.com/icons.svg#x"/>'],
    ['external filter', '<filter id="f"><feImage href="https://example.com/pixel"/></filter>'],
    ['CSS import', '<style>@import url("https://example.com/leak.css")</style>'],
    ['CSS declaration URL', '<style>.x{fill:url(http://127.0.0.1/private)}</style>'],
    ['SMIL href mutation', '<set attributeName="href" to="https://example.com/a.svg"/>']
  ])('removes the resource-capable element for %s', (_label, child) => {
    const svg = sanitizedSvg(`<svg>${child}<circle id="safe" r="1"/></svg>`)

    expect(svg.children).toHaveLength(1)
    expect(svg.querySelector('circle#safe')).not.toBeNull()
  })

  it.each([
    ['external fill', 'fill', 'url(https://example.com/paint.svg#p)'],
    ['localhost stroke', 'stroke', 'url(http://127.0.0.1/p)'],
    ['file mask', 'mask', 'url(file:///etc/passwd#m)'],
    ['data clip', 'clip-path', 'url(data:image/svg+xml,boom)'],
    ['external marker', 'marker-end', 'url(//example.com/marker.svg#m)'],
    ['escaped URL token', 'fill', 'u\\72l(https://example.com/p)'],
    ['comment-obscured URL token', 'stroke', 'u/**/rl(http://localhost/p)']
  ])('removes %s presentation references', (_label, name, value) => {
    const svg = sanitizedSvg(`<svg><path id="resource-node" ${name}="${value}" d="M0 0h1"/></svg>`)

    expect(svg.querySelector('#resource-node')?.hasAttribute(name)).toBe(false)
  })

  it('removes URI and CSS attributes even from otherwise safe elements', () => {
    const svg = sanitizedSvg(
      `<svg><circle id="resource-node" href="https://example.com" xlink:href="data:text/html,x" style="fill:url(http://localhost/p)" r="1"/></svg>`
    )

    const resourceNode = svg.querySelector('#resource-node')

    expect(resourceNode?.hasAttribute('href')).toBe(false)
    expect(resourceNode?.hasAttribute('xlink:href')).toBe(false)
    expect(resourceNode?.hasAttribute('style')).toBe(false)
  })

  it('still removes scripts, event handlers, and foreignObject', () => {
    const svg = sanitizedSvg(`<svg onload="alert(1)">
  <script>alert(1)</script>
  <foreignObject><button onclick="alert(1)">run</button></foreignObject>
  <circle id="safe" onclick="alert(1)" r="1"/>
</svg>`)

    expect(svg.hasAttribute('onload')).toBe(false)
    expect(svg.querySelector('script')).toBeNull()
    expect(svg.querySelector('foreignObject')).toBeNull()
    expect(svg.querySelector('#safe')?.hasAttribute('onclick')).toBe(false)
  })

  it('fails closed when sanitized content has no SVG root', () => {
    expect(sanitizeSvgMarkup('<p>not svg</p>')).toBe('')
  })
})
