import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LocalFilePreview, MarkdownPreview } from './preview-file'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

function installFileBridge(text: string, truncated = false) {
  const writeClipboard = vi.fn().mockResolvedValue(undefined)

  desktopWindow.hermesDesktop = {
    gitRoot: vi.fn().mockResolvedValue(null),
    readFileText: vi.fn().mockResolvedValue({
      byteSize: text.length,
      language: 'markdown',
      path: '/tmp/notes.md',
      text,
      truncated
    }),
    writeClipboard
  } as unknown as Window['hermesDesktop']

  return writeClipboard
}

const markdownTarget = {
  kind: 'file' as const,
  label: 'notes.md',
  language: 'markdown',
  path: '/tmp/notes.md',
  previewKind: 'text' as const,
  source: '/tmp/notes.md',
  url: 'file:///tmp/notes.md'
}

// Behavior tests for the .md file preview renderer: input markdown goes
// through normalizeFilePreviewMath -> Streamdown (+ KaTeX math plugin) and must
// come out as real rendered elements, matching what the chat transcript
// renderer produces. Guards the regression where the preview was a bare
// Streamdown pass with no math plugin and no table/img/a components.
describe('MarkdownPreview', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()

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

  it('renders external links to open in a new tab safely', () => {
    const { container } = render(<MarkdownPreview text={'[docs](https://example.com/docs)'} />)

    const anchor = container.querySelector('a')
    expect(anchor?.getAttribute('href')).toBe('https://example.com/docs')
    expect(anchor?.getAttribute('target')).toBe('_blank')
    expect(anchor?.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('copies the complete raw Markdown from the preview toolbar', async () => {
    const markdown = '# Heading\n\n**bold** and [link](https://example.com)\n'
    const writeClipboard = installFileBridge(markdown)

    render(<LocalFilePreview reloadKey={0} target={markdownTarget} />)
    const copy = await screen.findByRole('button', { name: 'Copy' })
    const edit = screen.getByRole('button', { name: 'Edit' })

    expect(copy.textContent).toBe('')
    expect(edit.textContent).toBe('')
    fireEvent.click(copy)

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith(markdown))
  })

  it.each([
    { text: '# Partial', truncated: true },
    { text: '', truncated: false }
  ])('does not offer a whole-file copy for truncated or empty Markdown', async ({ text, truncated }) => {
    installFileBridge(text, truncated)

    render(<LocalFilePreview reloadKey={0} target={markdownTarget} />)

    await screen.findByText('SOURCE')
    expect(screen.queryByRole('button', { name: 'Copy' })).toBeNull()
  })
})
