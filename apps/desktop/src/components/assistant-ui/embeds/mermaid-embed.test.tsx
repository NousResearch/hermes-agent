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
})
