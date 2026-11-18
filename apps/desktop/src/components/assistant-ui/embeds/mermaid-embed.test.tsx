import { act, cleanup, render, waitFor } from '@testing-library/react'
import type * as PeleLazy from 'pele/lazy'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Real Pele, spied: the assertions below are about what it actually emits.
const { renderAsync } = vi.hoisted(() => ({ renderAsync: vi.fn() }))

vi.mock('pele/lazy', async importOriginal => {
  const pele = await importOriginal<typeof PeleLazy>()
  renderAsync.mockImplementation(pele.renderAsync)

  return { ...pele, renderAsync }
})

import MermaidRenderer from './mermaid-embed'

const decodeImage = (img: Element | null) =>
  decodeURIComponent((img as HTMLImageElement | null)?.src.split(',')[1] ?? '')

const parseXml = (svg: string) => new DOMParser().parseFromString(svg, 'image/svg+xml')

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  window.document.documentElement.removeAttribute('style')
})

describe('MermaidRenderer', () => {
  it('reuses one render while isolating each mounted SVG in an image resource', async () => {
    const code = 'flowchart TD\n  accTitle: Request flow\n  accDescr: A sends data to B\n  A --> B'

    const { container } = render(
      <>
        <MermaidRenderer code={code} />
        <MermaidRenderer code={code} />
      </>
    )

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(2))

    const images = [...container.querySelectorAll('img')]
    expect(renderAsync).toHaveBeenCalledTimes(1)
    expect(container.querySelector('svg.pele')).toBeNull()
    expect(images[0]?.src).toMatch(/^data:image\/svg\+xml/)
    expect(images[1]?.src).toBe(images[0]?.src)
    expect(images.map(image => image.alt)).toEqual([
      'Request flow — A sends data to B',
      'Request flow — A sends data to B'
    ])
  })

  it('names a diagram without accTitle/accDescr generically', async () => {
    const { container } = render(<MermaidRenderer code={'flowchart TD\n  P --> Q'} />)

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))

    expect(container.querySelector('img')?.alt).toBe('Mermaid diagram')
  })

  it('does not name a diagram after a nested <title> (journey actor)', async () => {
    const { container } = render(
      <MermaidRenderer code={'journey\n  title Ship it\n  section Build\n    Code: 5: Alice, Bob'} />
    )

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))

    expect(decodeImage(container.querySelector('img'))).toContain('<title>Alice</title>')
    expect(container.querySelector('img')?.alt).not.toContain('Alice')
  })

  it('typesets $$TeX$$ labels as MathML instead of showing the source', async () => {
    // jsdom lays nothing out; give the measured MathML a browser-like size.
    const rect = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockReturnValue({ height: 18, width: 24 } as DOMRect)

    const { container } = render(
      <MermaidRenderer
        code={'flowchart LR\n  A["$$x^2$$"] --> B["$$E=mc^2$$"] --> T["$$y\\tag{1}$$"] --> C["$$\\frac{a$$"]'}
      />
    )

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))
    rect.mockRestore()

    const svg = decodeImage(container.querySelector('img'))
    const doc = parseXml(svg)

    expect(doc.querySelector('parsererror')).toBeNull()
    // Display mode, as mermaid typeset them: \tag is display-only.
    expect(doc.querySelectorAll('foreignObject.pele-math math')).toHaveLength(3)
    expect(svg).not.toContain('$$y\\tag{1}$$')
    expect(doc.querySelector('msup')?.textContent).toBe('x2')
    expect(svg).not.toContain('$$x^2$$')
    // Bare MathML: the app stylesheet enlarges `.katex`, and the SVG has none.
    expect(svg).not.toContain('class="katex"')
    // TeX KaTeX cannot parse stays readable as its source.
    expect(svg).toContain('\\frac{a')
  })

  it('sizes math labels in unzoomed pixels under the browser UI scale', async () => {
    // The browser-hosted app scales through CSS zoom on <html>: at 50% the
    // probe measures half size, but the SVG still draws at full size.
    const rect = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockReturnValue({ height: 9, width: 12 } as DOMRect)

    Object.defineProperty(HTMLElement.prototype, 'currentCSSZoom', { configurable: true, get: () => 0.5 })

    try {
      const { container } = render(<MermaidRenderer code={'flowchart LR\n  A["$$x^2$$"] --> B'} />)

      await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))

      const box = parseXml(decodeImage(container.querySelector('img'))).querySelector('foreignObject.pele-math')

      expect([box?.getAttribute('width'), box?.getAttribute('height')]).toEqual(['24', '18'])
    } finally {
      rect.mockRestore()
      delete (HTMLElement.prototype as { currentCSSZoom?: number }).currentCSSZoom
    }
  })

  it('resolves the theme once per repaint, not once per diagram', async () => {
    const { container } = render(
      <>
        {['A', 'B', 'C'].map(name => (
          <MermaidRenderer code={`flowchart LR\n  ${name}1 --> ${name}2`} key={name} />
        ))}
      </>
    )

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(3))
    const computed = vi.spyOn(window, 'getComputedStyle')

    await act(async () => {
      window.document.documentElement.style.setProperty('--dt-font-sans', 'Once Sans')
    })

    await waitFor(() => expect(renderAsync).toHaveBeenCalledTimes(6))
    const probeReads = computed.mock.calls.filter(([el]) => el instanceof HTMLSpanElement).length
    computed.mockRestore()

    // One read per distinct token for all three diagrams together.
    expect(probeReads).toBe(7)
  })

  it('keeps a label with <br/> and &nbsp; loadable as an image (#133089)', async () => {
    const { container } = render(<MermaidRenderer code={'graph TD\n  A["a<br/>b&nbsp;c"]'} />)

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))

    const doc = parseXml(decodeImage(container.querySelector('img')))

    expect(doc.documentElement.tagName).toBe('svg')
    expect(doc.querySelector('parsererror')).toBeNull()
    expect(doc.documentElement.textContent).toContain('b\u00a0c')
  })

  it('writes resolved app colors onto the SVG root, leaves unresolved ones to Pele, and draws no links', async () => {
    window.document.documentElement.style.setProperty('--dt-font-sans', 'Test Sans')

    // jsdom resolves no var(); stand in for the browser: only --dt-foreground
    // has a value, every other token falls through to the probe's sentinel.
    const realStyle = window.getComputedStyle

    const computed = vi.spyOn(window, 'getComputedStyle').mockImplementation((el, pseudo) => {
      if (el instanceof HTMLSpanElement) {
        const color = el.style.color.startsWith('var(--dt-foreground,') ? 'rgb(10, 20, 30)' : 'rgb(1, 2, 3)'

        return { color } as CSSStyleDeclaration
      }

      return realStyle(el, pseudo)
    })

    const { container } = render(
      <MermaidRenderer
        code={'flowchart LR\n  A --> B\n  click A href "https://example.com"\n  click B "javascript:alert(1)"'}
      />
    )

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))
    computed.mockRestore()

    const svg = decodeImage(container.querySelector('img'))
    const style = parseXml(svg).documentElement.getAttribute('style') ?? ''

    expect(style.startsWith('--pele-fg:rgb(10, 20, 30);')).toBe(true)
    expect(style.match(/--pele-[a-z-]+:/g)).toEqual(['--pele-fg:'])
    expect(renderAsync.mock.calls[0]?.[1]).toMatchObject({ fontFamily: 'Test Sans', images: false, links: false })
    expect(parseXml(svg).querySelector('a')).toBeNull()
    expect(svg).not.toContain('javascript:')
  })

  it('shows the source when Pele cannot draw the diagram', async () => {
    const code = 'flowchart TD\n  A -->'
    const { container } = render(<MermaidRenderer code={code} />)

    await waitFor(() => expect(container.querySelector('pre')?.className).toContain('text-foreground/90'))

    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('pre')?.textContent).toBe(code)
  })

  it('renders again when a theme repaint changes the resolved theme', async () => {
    const { container } = render(<MermaidRenderer code={'flowchart LR\n  A --> B'} />)

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(1))
    const before = container.querySelector('img')?.getAttribute('src')

    await act(async () => {
      window.document.documentElement.style.setProperty('--dt-font-sans', 'Repainted Sans')
    })

    await waitFor(() => {
      const src = container.querySelector('img')?.getAttribute('src')

      expect(src).toMatch(/^data:/)
      expect(src).not.toBe(before)
    })
    expect(renderAsync).toHaveBeenCalledTimes(2)
    expect(renderAsync.mock.calls[1]?.[1]).toMatchObject({ fontFamily: 'Repainted Sans' })
  })
})
