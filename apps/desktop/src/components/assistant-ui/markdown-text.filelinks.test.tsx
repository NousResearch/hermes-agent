import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent, MessageTextContent } from './markdown-text'

// Regression for #82140: a plain filesystem href in assistant markdown
// (`[report](/home/user/report.md)`) rendered as a bare dead anchor —
// file:// is blocked in the renderer, and on a remote gateway the path
// isn't on this disk at all. Such links must route through the preview
// pipeline (PreviewAttachment → normalizeOrLocalPreviewTarget), which
// resolves the path at VIEW time against the session's backend: local
// connections read the file directly, remote connections fetch it over the
// authenticated /api/fs bridge. Media-extension paths keep their inline
// player instead.
describe('MarkdownLink filesystem hrefs', () => {
  afterEach(cleanup)

  // A file named mid-sentence stays link text inside its paragraph — a block
  // card there splits the sentence in two. The link is the view-time door:
  // click resolves against the session's backend and opens the preview pane.
  it('renders an absolute file path link as an inline link, not a card', async () => {
    render(<MarkdownTextContent isRunning={false} text="Wrote it: [report](/home/user/report.md) for you." />)

    const link = await screen.findByRole('link', { name: 'report' })

    expect(link.getAttribute('title')).toBe('/home/user/report.md')
    expect(link.closest('p')?.textContent).toBe('Wrote it: report for you.')
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })

  it('routes file:// and ~/ links the same way', async () => {
    render(
      <MarkdownTextContent isRunning={false} text={'See [notes](file:///srv/data/notes.txt) and [todo](~/todo.md)'} />
    )

    expect(await screen.findByRole('link', { name: 'notes' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'todo' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })

  it('renders a MEDIA: tag mid-sentence as an inline link, and on its own line as a card', async () => {
    render(
      <MessageTextContent
        text={'Read MEDIA:/repo/README.md and the roadmap.\n\nMEDIA:/repo/roadmap.md\nMEDIA:/repo/notes.txt'}
      />
    )

    const inline = await screen.findByRole('link', { name: 'README.md' })

    expect(inline.closest('p')?.textContent).toBe('Read README.md and the roadmap.')
    // Back-to-back MEDIA: lines each keep their delivery card.
    expect(await screen.findAllByRole('button', { name: 'Open preview' })).toHaveLength(2)
  })

  it('renders a media player for a media-extension path link', async () => {
    const { container } = render(<MarkdownTextContent isRunning={false} text="[clip](/tmp/demo.mp4)" />)

    await waitFor(() => expect(container.querySelector('video')).not.toBeNull())
    expect(container.querySelector('a[href="/tmp/demo.mp4"]')).toBeNull()
  })

  it('leaves anchors and relative links out of the preview pipeline', () => {
    render(
      <MarkdownTextContent
        isRunning={false}
        text={'[frag](#section-2) and [rel](docs/guide.md) and [site](https://example.com)'}
      />
    )

    // Fragment anchors survive untouched; relative links are NOT rewritten
    // (they keep Streamdown's pre-existing handling) — neither gains a
    // preview affordance.
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
    expect(document.querySelector('a[href="#section-2"]')).not.toBeNull()
  })
})
