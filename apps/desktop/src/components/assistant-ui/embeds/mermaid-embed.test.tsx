import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { initialize, renderMermaid } = vi.hoisted(() => ({
  initialize: vi.fn(),
  renderMermaid: vi.fn(async () => ({
    svg: '<svg id="shared"><title>Request flow</title><desc>A sends data to B</desc><defs><marker id="arrow" /></defs><path marker-end="url(#arrow)" /></svg>'
  }))
}))

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

  it('repairs a label mermaid returned as HTML, so the image stays loadable', async () => {
    // `A["a<br/>b&nbsp;c"]` comes back from mermaid with an open `<br>` and an
    // `&nbsp;` entity — malformed XML, so the data: URI fails to decode and the
    // message shows a broken image with the alt text instead of the diagram
    // (#133089).
    renderMermaid.mockResolvedValueOnce({
      svg: '<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 260.34375 70"><foreignObject width="19.109375" height="48"><div xmlns="http://www.w3.org/1999/xhtml"><span class="nodeLabel"><p>a<br>b&nbsp;c</p></span></div></foreignObject></svg>'
    })

    const { container } = render(<MermaidRenderer code='graph TD; A["a<br/>b&nbsp;c"]' />)

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))

    const src = decodeURIComponent([...container.querySelectorAll('img')][0]?.src.split(',')[1] ?? '')
    const doc = new DOMParser().parseFromString(src, 'image/svg+xml')

    expect(doc.documentElement.tagName).toBe('svg')
    expect(doc.querySelector('p')?.textContent).toBe('ab c')
  })
})
