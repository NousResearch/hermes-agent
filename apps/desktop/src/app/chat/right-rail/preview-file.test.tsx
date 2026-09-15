import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { friendlyPreviewError, MarkdownPreview } from './preview-file'

// Behavior tests for the .md file preview renderer: input markdown goes
// through normalizeFilePreviewMath -> Streamdown (+ KaTeX math plugin) and must
// come out as real rendered elements, matching what the chat transcript
// renderer produces. Guards the regression where the preview was a bare
// Streamdown pass with no math plugin and no table/img/a components.
describe('MarkdownPreview', () => {
  afterEach(() => {
    cleanup()
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
})

// Guards #105750: a deleted/moved file surfaced Electron's "Error invoking
// remote method 'hermes:readFileDataUrl'" wrapper verbatim in the pane.
describe('friendlyPreviewError', () => {
  const fileGoneBody = 'This file may have been moved, renamed, or deleted.'

  it('turns the wrapped IPC missing-file failure into friendly copy', () => {
    const message = friendlyPreviewError(
      new Error("Error invoking remote method 'hermes:readFileDataUrl': Error: File preview failed: file does not exist."),
      fileGoneBody
    )

    expect(message).toBe(fileGoneBody)
  })

  it('recognizes raw ENOENT from a file deleted between stat and read', () => {
    expect(friendlyPreviewError(new Error("ENOENT: no such file or directory, open '/tmp/gone.png'"), fileGoneBody)).toBe(
      fileGoneBody
    )
  })

  it('strips the invoke wrapper but keeps readable causes', () => {
    const message = friendlyPreviewError(
      new Error("Error invoking remote method 'hermes:readFileDataUrl': Error: File preview failed: file is too large (20971520 bytes; limit 16777216 bytes)."),
      fileGoneBody
    )

    expect(message).toBe('File preview failed: file is too large (20971520 bytes; limit 16777216 bytes).')
  })

  it('passes plain errors and non-Error values through', () => {
    expect(friendlyPreviewError(new Error('Timed out'), fileGoneBody)).toBe('Timed out')
    expect(friendlyPreviewError('boom', fileGoneBody)).toBe('boom')
  })
})
