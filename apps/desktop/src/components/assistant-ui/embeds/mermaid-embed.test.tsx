import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { initialize, renderMermaid } = vi.hoisted(() => ({
  initialize: vi.fn(),
  renderMermaid: vi.fn(async () => ({
    svg: '<svg id="shared"><title>Request flow</title><desc>A sends data to B</desc><defs><marker id="arrow" /></defs><path marker-end="url(#arrow)" /></svg>'
  }))
}))

// Real mermaid 11 strict-mode output shape: node labels live in a
// foreignObject, and the strict-mode DOMPurify pass rewrote the label's
// self-closing <br/> into the HTML-dequalified bare <br>.
const BR_SVG =
  '<svg id="mmd-br" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width: 200px;" viewBox="0 0 200 60" role="graphics-document document"><g><foreignObject width="200" height="60"><div xmlns="http://www.w3.org/1999/xhtml"><p>A<br>B</p></div></foreignObject></g></svg>'

vi.mock('mermaid', () => ({
  default: {
    initialize,
    render: renderMermaid
  }
}))

vi.mock('./use-is-dark', () => ({ useIsDark: () => false }))

import MermaidRenderer from './mermaid-embed'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('MermaidRenderer', () => {
  it('reuses one render while isolating each mounted SVG in an image resource', async () => {
    const { container } = render(
      <>
        <MermaidRenderer code="graph TD; A-->B" />
        <MermaidRenderer code="graph TD; A-->B" />
      </>
    )

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(2))

    const images = [...container.querySelectorAll('img')]
    expect(renderMermaid).toHaveBeenCalledTimes(1)
    expect(container.querySelector('svg#shared')).toBeNull()
    expect(images[0]?.src).toMatch(/^data:image\/svg\+xml/)
    expect(images[1]?.src).toBe(images[0]?.src)
    expect(images.map(image => image.alt)).toEqual([
      'Request flow — A sends data to B',
      'Request flow — A sends data to B'
    ])
    expect(decodeURIComponent(images[0]?.src.split(',')[1] ?? '')).toContain('marker-end="url(#arrow)"')
  })

  it('re-closes void <br> that strict-mode sanitisation dequalified before the img round-trip', async () => {
    renderMermaid.mockImplementationOnce(async () => ({ svg: BR_SVG }))

    const { container } = render(<MermaidRenderer code="graph TD; A[line-break] --- B" />)

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))

    const svg = decodeURIComponent(container.querySelector('img')?.src.split(',')[1] ?? '')

    // The data URL is decoded by the <img> as XML, so every void tag must be
    // closed again or the whole diagram fails to load. (The size rewriter
    // re-serialises the XML, which emits <br /> with a space.)
    expect(svg).toMatch(/<br\s*\/>/)
    expect(svg).not.toMatch(/<br\s*(?!\/)>/)
    // The 100% width recomputes only if the SVG still parses as XML — the
    // regression also manifested there (parsererror left width at 100%).
    expect(svg).toContain('width="200"')
    expect(svg).not.toContain('width="100%"')
  })
})
