import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

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

  it('renders external links to open in a new tab safely', () => {
    const { container } = render(<MarkdownPreview text={'[docs](https://example.com/docs)'} />)

    const anchor = container.querySelector('a')
    expect(anchor?.getAttribute('href')).toBe('https://example.com/docs')
    expect(anchor?.getAttribute('target')).toBe('_blank')
    expect(anchor?.getAttribute('rel')).toBe('noopener noreferrer')
  })
})
