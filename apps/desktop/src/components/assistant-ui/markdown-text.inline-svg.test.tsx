import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const SIMPLE_SVG = `<svg id="raw-inline-svg" width="48" height="48" viewBox="0 0 24 24">
  <path d="M13 2L6 13h5l-1 9 8-12h-5z" />
</svg>`

function renderMarkdown(text: string, isRunning = false) {
  return render(<MarkdownTextContent isRunning={isRunning} text={text} />)
}

describe('MarkdownTextContent raw SVG', () => {
  afterEach(cleanup)

  it('renders a balanced unfenced SVG through the production markdown surface', async () => {
    const { container } = renderMarkdown(`Before\n\n${SIMPLE_SVG}\n\nAfter`)

    await waitFor(() => expect(container.querySelector('svg#raw-inline-svg')).not.toBeNull())
    expect(container.textContent).toContain('Before')
    expect(container.textContent).toContain('After')
  })

  it('reprocesses text after a terminated quoted fence and renders a later nested-quote SVG', async () => {
    const text = [
      '> ```html',
      '> quoted code',
      'root prose',
      ...SIMPLE_SVG.split('\n').map(line => `> > ${line}`)
    ].join('\n')

    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.querySelector('svg#raw-inline-svg')).not.toBeNull())
    expect(container.textContent).toContain('root prose')

    const svg = container.querySelector('svg#raw-inline-svg')
    const innerQuote = svg?.closest('blockquote')

    expect(innerQuote?.parentElement?.closest('blockquote')).not.toBeNull()
  })

  it('keeps and sanitizes a generated SVG fence inside nested list and quote containers', async () => {
    const dangerous = SIMPLE_SVG.replace(
      '<svg id="raw-inline-svg"',
      '<svg id="nested-container-svg" onload="alert(1)"'
    ).replace('<path ', '<script>alert(1)</script>\n  <path onclick="alert(1)" ')

    const text = ['- outer', ...dangerous.split('\n').map(line => `  > ${line}`)].join('\n')
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.querySelector('svg#nested-container-svg')).not.toBeNull())

    const svg = container.querySelector('svg#nested-container-svg')

    expect(svg?.closest('blockquote')?.closest('li')).not.toBeNull()
    expect(svg?.hasAttribute('onload')).toBe(false)
    expect(svg?.querySelector('script')).toBeNull()
    expect(svg?.querySelector('path')?.hasAttribute('onclick')).toBe(false)
  })

  it('sanitizes active content and dangerous URLs from an unfenced SVG', async () => {
    const dangerous = `<svg id="sanitized-inline-svg" viewBox="0 0 24 24" onload="alert(1)">
  <script>alert(1)</script>
  <foreignObject><button id="foreign-button" onclick="alert(1)">Run</button></foreignObject>
  <a id="dangerous-link" href="javascript:alert(1)" xlink:href="data:text/html,boom">
    <circle id="safe-circle" cx="12" cy="12" r="10" onclick="alert(1)" />
  </a>
</svg>`

    const { container } = renderMarkdown(dangerous)

    await waitFor(() => expect(container.querySelector('svg#sanitized-inline-svg')).not.toBeNull())

    const svg = container.querySelector('svg#sanitized-inline-svg')
    const link = container.querySelector('#dangerous-link')
    const circle = container.querySelector('#safe-circle')

    expect(svg?.hasAttribute('onload')).toBe(false)
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('foreignObject')).toBeNull()
    expect(container.querySelector('#foreign-button')).toBeNull()
    expect(link?.hasAttribute('href')).toBe(false)
    expect(link?.hasAttribute('xlink:href')).toBe(false)
    expect(circle?.hasAttribute('onclick')).toBe(false)
  })

  it.each([
    ['fenced code', `\`\`\`text\n${SIMPLE_SVG}\n\`\`\``],
    [
      'blockquoted fenced code',
      `> \`\`\`html\n${SIMPLE_SVG.split('\n')
        .map(line => `> ${line}`)
        .join('\n')}\n> \`\`\``
    ],
    [
      'nested-blockquoted fenced code',
      `> > ~~~~html\n${SIMPLE_SVG.split('\n')
        .map(line => `> > ${line}`)
        .join('\n')}\n> > ~~~~`
    ],
    [
      'list-nested fenced code',
      `- ~~~html\n${SIMPLE_SVG.split('\n')
        .map(line => `  ${line}`)
        .join('\n')}\n  ~~~`
    ],
    [
      'nested-list blockquoted fenced code',
      `> - 1. ~~~~html\n${SIMPLE_SVG.split('\n')
        .map(line => `>      ${line}`)
        .join('\n')}\n>      ~~~~`
    ],
    [
      'nested-list continuation fenced code',
      `- outer\n  - inner\n      ~~~~html\n${SIMPLE_SVG.split('\n')
        .map(line => `    ${line}`)
        .join('\n')}\n    ~~~~`
    ],
    ['inline code', `Keep this as code: \`${SIMPLE_SVG.replaceAll('\n', ' ')}\``],
    [
      'indented code',
      SIMPLE_SVG.split('\n')
        .map(line => `    ${line}`)
        .join('\n')
    ],
    [
      'blockquoted indented code',
      SIMPLE_SVG.split('\n')
        .map(line => `>     ${line}`)
        .join('\n')
    ]
  ])('keeps SVG inside %s as code', async (_label, text) => {
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.textContent).toContain('<svg'))
    expect(container.querySelector('code')?.textContent).toContain('<svg')
    expect(container.querySelector('svg#raw-inline-svg')).toBeNull()
  })

  it('keeps inline-code SVG inert after a dangling delimiter in an earlier paragraph', async () => {
    const dangerous = SIMPLE_SVG.replace(
      '<svg id="raw-inline-svg"',
      '<svg id="inline-code-security-svg" onload="alert(1)"'
    ).replace('<path ', '<script>alert(1)</script><path onclick="alert(1)" ')

    const text = ['`dangling opener', '', `\`${dangerous.replaceAll('\n', ' ')}\``].join('\n')
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.querySelector('code')?.textContent).toContain('<svg'))
    expect(container.querySelector('svg#inline-code-security-svg')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })

  it.each([
    [
      'four-column unordered-list continuation',
      ['- item', ...SIMPLE_SVG.split('\n').map(line => `    ${line}`)].join('\n')
    ],
    [
      'five-column ordered-list continuation',
      ['123. item', ...SIMPLE_SVG.split('\n').map(line => `     ${line}`)].join('\n')
    ]
  ])('renders raw SVG from a %s', async (_label, text) => {
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.querySelector('svg#raw-inline-svg')).not.toBeNull())
    expect(container.querySelector('svg#raw-inline-svg')?.closest('li')).not.toBeNull()
  })

  it('renders a root SVG after an unterminated quoted SVG container', async () => {
    const rootSvg = SIMPLE_SVG.replace('raw-inline-svg', 'root-after-unterminated-quote')
    const text = ['> <svg id="unterminated-quote"><path/>', '', rootSvg].join('\n')
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.querySelector('svg#root-after-unterminated-quote')).not.toBeNull())
    expect(container.querySelector('svg#unterminated-quote')).toBeNull()
  })

  it('does not render a root SVG closed from a later blockquote after a blank boundary', async () => {
    const text = '<svg id="borrowed-root"><path/>\n\n> </svg>'
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.querySelector('blockquote')).not.toBeNull())
    expect(container.querySelector('svg#borrowed-root')).toBeNull()
    expect(container.querySelectorAll('blockquote')).toHaveLength(1)
  })

  it.each([
    {
      containerSelector: 'blockquote',
      id: 'quoted-sibling',
      label: 'blockquote',
      sibling: SIMPLE_SVG.replace('raw-inline-svg', 'quoted-sibling')
        .split('\n')
        .map(line => `> ${line}`)
        .join('\n')
    },
    {
      containerSelector: 'li',
      id: 'list-sibling',
      label: 'list',
      sibling: SIMPLE_SVG.replace('raw-inline-svg', 'list-sibling')
        .split('\n')
        .map((line, index) => `${index === 0 ? '- ' : '  '}${line}`)
        .join('\n')
    }
  ])('renders a sibling SVG after a root-to-$label block transition', async ({ containerSelector, id, sibling }) => {
    const text = ['<svg id="abandoned-root"><path/>', '', sibling].join('\n')
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.querySelector(`svg#${id}`)).not.toBeNull())
    expect(container.querySelector('svg#abandoned-root')).toBeNull()
    expect(container.querySelector(`svg#${id}`)?.closest(containerSelector)).not.toBeNull()
    expect(container.querySelectorAll('svg[id]')).toHaveLength(1)
  })

  it('preserves the existing fenced SVG renderer', async () => {
    const { container } = renderMarkdown(`\`\`\`svg\n${SIMPLE_SVG}\n\`\`\``)

    await waitFor(() => expect(container.querySelector('svg#raw-inline-svg')).not.toBeNull())
  })

  it('does not cross a code fence to close malformed raw SVG', async () => {
    const text = `<svg id="malformed-inline-svg" viewBox="0 0 10 10">\n<path d="M0 0h10v10z">\n\n\`\`\`text\n</svg>\n\`\`\``
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.textContent).toContain('</svg>'))
    expect(container.querySelector('svg#malformed-inline-svg')).toBeNull()
  })

  it('does not use indented code to close malformed raw SVG', async () => {
    const text = '<svg id="malformed-indented-svg"><path d="M0 0">\n\n    </svg>'
    const { container } = renderMarkdown(text)

    await waitFor(() => expect(container.textContent).toContain('</svg>'))
    expect(container.querySelector('svg#malformed-indented-svg')).toBeNull()
  })

  it('keeps arbitrary raw HTML inert while enabling only SVG', async () => {
    const { container } = renderMarkdown(
      `<button id="raw-html-button" onclick="alert(1)">Unsafe</button>\n\n${SIMPLE_SVG}`
    )

    await waitFor(() => expect(container.querySelector('svg#raw-inline-svg')).not.toBeNull())
    expect(container.querySelector('#raw-html-button')).toBeNull()
  })

  it('waits for the closing tag while streaming and renders once balanced', async () => {
    const incomplete = SIMPLE_SVG.slice(0, SIMPLE_SVG.lastIndexOf('</svg>'))
    const view = renderMarkdown(incomplete, true)

    expect(view.container.querySelector('svg#raw-inline-svg')).toBeNull()

    view.rerender(<MarkdownTextContent isRunning text={SIMPLE_SVG} />)
    await waitFor(() => expect(view.container.querySelector('svg#raw-inline-svg')).not.toBeNull())
  })

  const resourceSvg = `<svg id="resource-policy-svg" viewBox="0 0 40 40">
  <defs>
    <linearGradient id="safe-gradient"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>
    <style>@import url("http://localhost/private"); .leak { fill: url(data:image/svg+xml,boom) }</style>
    <filter id="external-filter"><feImage href="file:///etc/passwd"/></filter>
  </defs>
  <rect id="safe-gradient-rect" width="20" height="20" fill="url(#safe-gradient)"/>
  <text id="safe-text" x="1" y="30">safe</text>
  <image id="remote-image" href="https://example.com/tracker.svg"/>
  <image id="localhost-image" href="http://127.0.0.1:8080/private"/>
  <use id="external-use" href="//example.com/icons.svg#secret"/>
  <path id="external-fill" d="M0 0h1" fill="url(https://example.com/paint.svg#p)"/>
  <path id="data-stroke" d="M0 1h1" stroke="url(data:image/svg+xml,boom)"/>
  <path id="external-filter-target" d="M0 2h1" filter="url(file:///etc/passwd#f)"/>
  <circle id="styled-circle" class="leak" style="background:url(http://localhost/private)" cx="30" cy="10" r="5"/>
</svg>`

  it.each([
    ['raw SVG', resourceSvg],
    ['fenced SVG', `\`\`\`svg\n${resourceSvg}\n\`\`\``]
  ])('forbids SVG resource loads from %s while retaining safe content', async (_label, markdown) => {
    const { container } = renderMarkdown(markdown)

    await waitFor(() => expect(container.querySelector('svg#resource-policy-svg')).not.toBeNull())

    expect(container.querySelector('#safe-gradient')).not.toBeNull()
    expect(container.querySelector('#safe-gradient-rect')?.getAttribute('fill')).toBe('url(#safe-gradient)')
    expect(container.querySelector('#safe-text')?.textContent).toBe('safe')
    expect(container.querySelector('style')).toBeNull()
    expect(container.querySelector('filter')).toBeNull()
    expect(container.querySelector('feImage')).toBeNull()
    expect(container.querySelector('image')).toBeNull()
    expect(container.querySelector('use')).toBeNull()
    expect(container.querySelector('#external-fill')?.hasAttribute('fill')).toBe(false)
    expect(container.querySelector('#data-stroke')?.hasAttribute('stroke')).toBe(false)
    expect(container.querySelector('#external-filter-target')?.hasAttribute('filter')).toBe(false)
    expect(container.querySelector('#styled-circle')?.hasAttribute('style')).toBe(false)
  })
})
