import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import * as desktopFs from '@/lib/desktop-fs'
import { setAlwaysExternalLinks } from '@/store/external-links'
import { $previewTabs } from '@/store/preview'

import { MarkdownPreview } from './preview-file'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

// Behavior tests for the .md file preview renderer: input markdown goes
// through normalizeFilePreviewMath -> Streamdown (+ KaTeX math plugin) and must
// come out as real rendered elements, matching what the chat transcript
// renderer produces. Guards the regression where the preview was a bare
// Streamdown pass with no math plugin and no table/img/a components.
describe('MarkdownPreview', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    cleanup()

    if (initialHermesDesktop) {
      desktopWindow.hermesDesktop = initialHermesDesktop
    } else {
      delete desktopWindow.hermesDesktop
    }
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
    $previewTabs.set([])
    setAlwaysExternalLinks(false)
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
    $previewTabs.set([])

    const hash = window.location.hash
    const scroll = vi.fn()

    HTMLElement.prototype.scrollIntoView = scroll

    const { container } = render(
      <MarkdownPreview text={'## Managed Tiered KV Cache\n\n[jump](#managed-tiered-kv-cache)'} />
    )

    const heading = container.querySelector('#managed-tiered-kv-cache')

    expect(heading?.tagName).toBe('H2')
    fireEvent.click(screen.getByRole('link', { name: 'jump' }))
    expect(scroll).toHaveBeenCalledTimes(1)
    expect(window.location.hash).toBe(hash)
    expect($previewTabs.get()).toEqual([])
  })

  it('scrolls to a heading whose TOC link uses a different Unicode normalization', () => {
    const scroll = vi.fn()

    HTMLElement.prototype.scrollIntoView = scroll

    render(<MarkdownPreview text={'## \u30D3\n\n[jump](#\u30D2\u3099)'} />)

    fireEvent.click(screen.getByRole('link', { name: 'jump' }))
    expect(scroll).toHaveBeenCalledTimes(1)
  })

  it('opens a relative markdown link as the sibling file', async () => {
    $previewTabs.set([])
    render(<MarkdownPreview filePath="/vault/notes/index.md" text={'[next](../other.md)'} />)

    fireEvent.click(screen.getByRole('link', { name: 'next' }))

    await waitFor(() => {
      expect($previewTabs.get().some(tab => tab.target.kind === 'file' && tab.target.path === '/vault/other.md')).toBe(
        true
      )
    })
  })

  it('does not navigate a scriptable link', () => {
    $previewTabs.set([])

    const { container } = render(<MarkdownPreview text={'[bad](javascript:alert(1))'} />)

    expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
    expect($previewTabs.get()).toEqual([])
  })

  it('resolves relative SVG images against each markdown file without reusing the first processor', async () => {
    const readFileDataUrl = vi.fn(async (path: string) =>
      path.startsWith('/one/') ? 'data:image/svg+xml;base64,b25l' : 'data:image/svg+xml;base64,dHdv'
    )

    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    const view = render(
      <MarkdownPreview filePath="/one/docs/report.md" fsCacheKey="local:" text="![chart](images/chart.svg)" />
    )

    await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toContain('b25l'))

    view.rerender(
      <MarkdownPreview filePath="/two/docs/report.md" fsCacheKey="local:" text="![chart](images/chart.svg)" />
    )

    await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toContain('dHdv'))
    expect(readFileDataUrl).toHaveBeenNthCalledWith(1, '/one/docs/images/chart.svg', '/one/docs/report.md', 2_097_152)
    expect(readFileDataUrl).toHaveBeenNthCalledWith(2, '/two/docs/images/chart.svg', '/two/docs/report.md', 2_097_152)
  })

  it('reloads same-path images when the filesystem connection identity changes', async () => {
    const readFileDataUrl = vi
      .fn<() => Promise<string>>()
      .mockResolvedValueOnce('data:image/svg+xml;base64,b25l')
      .mockResolvedValueOnce('data:image/svg+xml;base64,dHdv')

    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    const view = render(
      <MarkdownPreview filePath="/docs/report.md" fsCacheKey="remote:a" text="![chart](images/chart.svg)" />
    )

    await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toContain('b25l'))

    view.rerender(
      <MarkdownPreview filePath="/docs/report.md" fsCacheKey="remote:b" text="![chart](images/chart.svg)" />
    )

    await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toContain('dHdv'))
    expect(readFileDataUrl).toHaveBeenCalledTimes(2)
  })

  it('does not trust source-authored internal metadata or read parent paths', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:image/svg+xml;base64,c2VjcmV0')
    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    const text = [
      '<img alt="forged" src="https://example.com/chart.svg" data-hermes-file-image-path="/etc/passwd">',
      '',
      '![outside](../outside.svg)'
    ].join('\n')

    const { container } = render(
      <MarkdownPreview filePath="/workspace/docs/report.md" fsCacheKey="local:" text={text} />
    )

    await waitFor(() => expect(container.querySelector('img[alt="forged"]')).not.toBeNull())
    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(container.querySelector('img[alt="forged"]')?.getAttribute('src')).toBe('https://example.com/chart.svg')
    expect(container.querySelector('img[alt="outside"]')).toBeNull()
  })

  it('surfaces relative image read failures instead of leaving a loading state', async () => {
    const readFileDataUrl = vi.fn(async () => {
      throw new Error('missing')
    })

    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    render(
      <MarkdownPreview filePath="/workspace/docs/report.md" fsCacheKey="local:" text="![chart](images/chart.svg)" />
    )

    expect(await screen.findByText(/Couldn't load chart\.svg/)).toBeTruthy()
    expect(screen.queryByText(/Loading chart\.svg/)).toBeNull()
    expect(screen.queryByRole('button', { name: 'Open image' })).toBeNull()
  })

  it('surfaces scoped image decode failures without an unscoped open action', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:image/svg+xml;base64,bm90LXJlYWxseS1hbi1pbWFnZQ==')
    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    render(
      <MarkdownPreview filePath="/workspace/docs/report.md" fsCacheKey="local:" text="![chart](images/chart.svg)" />
    )

    const image = await screen.findByRole('img', { name: 'chart' })
    fireEvent.error(image)

    expect(await screen.findByText(/Couldn't load chart\.svg/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Open image' })).toBeNull()
  })

  it('disposes in-flight image loads when filesystem authority changes', async () => {
    let resolveFirst!: (value: string) => void

    const first = new Promise<string>(resolve => {
      resolveFirst = resolve
    })

    const readFileDataUrl = vi
      .fn<() => Promise<string>>()
      .mockReturnValueOnce(first)
      .mockResolvedValueOnce('data:image/svg+xml;base64,bmV3')

    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    const view = render(
      <MarkdownPreview filePath="/docs/report.md" fsCacheKey="remote:a" text="![chart](images/chart.svg)" />
    )

    await waitFor(() => expect(readFileDataUrl).toHaveBeenCalledTimes(1))
    view.rerender(
      <MarkdownPreview filePath="/docs/report.md" fsCacheKey="remote:b" text="![chart](images/chart.svg)" />
    )

    await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toContain('bmV3'))
    resolveFirst('data:image/svg+xml;base64,b2xk')
    await first
    expect(view.container.querySelector('img')?.getAttribute('src')).not.toContain('b2xk')
  })

  it('disposes saturated image work when the same document content changes', async () => {
    let resolveFirst!: (value: string) => void
    let resolveSecond!: (value: string) => void

    const first = new Promise<string>(resolve => {
      resolveFirst = resolve
    })

    const second = new Promise<string>(resolve => {
      resolveSecond = resolve
    })

    const readFileDataUrl = vi
      .fn<(path: string, relativeToFile?: string, maxBytes?: number) => Promise<string>>()
      .mockReturnValueOnce(first)
      .mockReturnValueOnce(second)
      .mockResolvedValueOnce('data:image/svg+xml;base64,bmV3')

    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    const view = render(
      <MarkdownPreview
        filePath="/docs/report.md"
        fsCacheKey="remote:a"
        text={'![one](images/one.svg)\n![two](images/two.svg)'}
      />
    )

    await waitFor(() => expect(readFileDataUrl).toHaveBeenCalledTimes(2))

    view.rerender(<MarkdownPreview filePath="/docs/report.md" fsCacheKey="remote:a" text="![new](images/new.svg)" />)
    await screen.findByText(/Loading new\.svg/)
    expect(readFileDataUrl).toHaveBeenCalledTimes(2)

    view.rerender(
      <MarkdownPreview filePath="/docs/report.md" fsCacheKey="remote:a" text="![latest](images/latest.svg)" />
    )
    await screen.findByText(/Loading latest\.svg/)
    expect(readFileDataUrl).toHaveBeenCalledTimes(2)

    resolveFirst('data:image/svg+xml;base64,b2xkLW9uZQ==')
    await first
    await waitFor(() => expect(readFileDataUrl).toHaveBeenCalledTimes(3))
    expect(readFileDataUrl.mock.calls[2]?.[0]).toBe('/docs/images/latest.svg')
    await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toContain('bmV3'))

    resolveSecond('data:image/svg+xml;base64,b2xkLXR3bw==')
    await second
    expect(view.container.querySelector('img')?.getAttribute('src')).not.toContain('b2xk')
  })

  it('treats an empty relative image response as a load failure', async () => {
    const readFileDataUrl = vi.fn(async () => '')

    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    render(
      <MarkdownPreview filePath="/workspace/docs/report.md" fsCacheKey="local:" text="![chart](images/chart.svg)" />
    )

    expect(await screen.findByText(/Couldn't load chart\.svg/)).toBeTruthy()
    expect(screen.queryByText(/Loading chart\.svg/)).toBeNull()
  })

  it('rejects a non-image data URL returned for a relative image', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,bm90IGFuIGltYWdl')

    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']

    render(
      <MarkdownPreview filePath="/workspace/docs/report.md" fsCacheKey="local:" text="![chart](images/chart.svg)" />
    )

    expect(await screen.findByText(/Couldn't load chart\.svg/)).toBeTruthy()
    expect(screen.queryByRole('img', { name: 'chart' })).toBeNull()
  })

  it('keeps scoped images usable across StrictMode effect replay', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:image/svg+xml;base64,PHN2Zy8+')
    desktopWindow.hermesDesktop = { readFileDataUrl } as unknown as Window['hermesDesktop']
    const view = render(
      <StrictMode>
        <MarkdownPreview filePath="/docs/report.md" fsCacheKey="local:" text="![chart](chart.svg)" />
      </StrictMode>
    )
    await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toContain('PHN2Zy8+'))
    expect(readFileDataUrl).toHaveBeenCalledTimes(1)
  })
  it('binds relative images to the document session and rotates the loader on session changes', async () => {
    const read = vi.spyOn(desktopFs, 'readDesktopFileDataUrl').mockResolvedValue(
      'data:image/svg+xml;base64,PHN2Zy8+'
    )
    const { rerender } = render(
      <MarkdownPreview filePath="/workspace/report.md" sessionId="session-a" text="![chart](chart.svg)" />
    )
    await waitFor(() => expect(read).toHaveBeenCalledWith('/workspace/chart.svg', {
      maxBytes: 2_097_152, relativeToFile: '/workspace/report.md', sessionId: 'session-a'
    }))
    read.mockClear()
    rerender(<MarkdownPreview filePath="/workspace/report.md" sessionId="session-b" text="![chart](chart.svg)" />)
    await waitFor(() => expect(read).toHaveBeenCalledWith('/workspace/chart.svg', {
      maxBytes: 2_097_152, relativeToFile: '/workspace/report.md', sessionId: 'session-b'
    }))
  })

})
