import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setAlwaysExternalLinks } from '@/store/external-links'
import { $previewTabs } from '@/store/preview'

import { MarkdownPreview } from './preview-file'

// Behavior tests for the .md file preview renderer: input markdown goes
// through normalizeFilePreviewMath -> Streamdown (+ KaTeX math plugin) and must
// come out as real rendered elements, matching what the chat transcript
// renderer produces. Guards the regression where the preview was a bare
// Streamdown pass with no math plugin and no table/img/a components.
describe('MarkdownPreview', () => {
  afterEach(() => {
    cleanup()
    $previewTabs.set([])
    setAlwaysExternalLinks(false)
  })

  it('renders block and inline math through KaTeX', () => {
    // KaTeX marks its output; raw "$" delimiters must be gone.
    const { container } = render(
      <MarkdownPreview
        text={'Formula:\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}\n$$\n\nInline $a^2 + b^2 = c^2$ too.'}
      />
    )

    expect(container.querySelector('.katex')).not.toBeNull()
    expect(screen.queryByText(/\$\$/)).toBeNull()
  })

  it('renders GFM tables with header and body cells', () => {
    const { container } = render(<MarkdownPreview text={'| h1 | h2 |\n| --- | --- |\n| a | b |'} />)

    const table = container.querySelector('table')
    expect(table).not.toBeNull()
    expect(table?.querySelector('thead th')?.textContent).toBe('h1')
    expect(table?.querySelector('tbody td')?.textContent).toBe('a')
  })

  it('renders images with alt text', () => {
    const { container } = render(<MarkdownPreview text={'![a chart](https://example.com/chart.png)'} />)

    const img = container.querySelector('img')
    expect(img?.getAttribute('alt')).toBe('a chart')
    expect(img?.getAttribute('src')).toBe('https://example.com/chart.png')
  })

  it('opens an https link in the in-app browser instead of a blank window', async () => {
    render(<MarkdownPreview text={'[docs](https://example.com/docs)'} />)

    const anchor = screen.getByRole('link', { name: 'docs' })
    const click = new MouseEvent('click', { bubbles: true, cancelable: true })

    anchor.dispatchEvent(click)
    expect(click.defaultPrevented).toBe(true)

    await waitFor(() => {
      expect(
        $previewTabs.get().some(tab => tab.target.kind === 'url' && tab.target.url === 'https://example.com/docs')
      ).toBe(true)
    })
  })

  it('scrolls a table-of-contents link to its heading without changing the app route', () => {
    const hash = window.location.hash
    const scroll = vi.fn()

    HTMLElement.prototype.scrollIntoView = scroll

    const { container } = render(
      <MarkdownPreview text={'## Managed Tiered KV Cache\n\n[jump](#managed-tiered-kv-cache)'} />
    )

    const heading = container.querySelector('#managed-tiered-kv-cache')

    expect(heading?.tagName).toBe('H2')
    fireEvent.click(screen.getByRole('link', { name: 'jump' }))
    expect(scroll).toHaveBeenCalled()
    expect(window.location.hash).toBe(hash)
    expect($previewTabs.get()).toEqual([])
  })

  it('opens a relative markdown link as the sibling file', async () => {
    render(<MarkdownPreview filePath="/vault/notes/index.md" text={'[next](../other.md)'} />)

    fireEvent.click(screen.getByRole('link', { name: 'next' }))

    await waitFor(() => {
      expect($previewTabs.get().some(tab => tab.target.kind === 'file' && tab.target.path === '/vault/other.md')).toBe(
        true
      )
    })
  })

  it('does not navigate a scriptable link', () => {
    const { container } = render(<MarkdownPreview text={'[bad](javascript:alert(1))'} />)

    expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
    expect(container.textContent).toMatch(/blocked/i)
    expect($previewTabs.get()).toEqual([])
  })
})
