import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { isMarkdownDocumentPath, mediaMarkdownHref } from '@/lib/media'

import { MarkdownTextContent } from './markdown-text'

// Regression for #84951: a `.md` delivered via MEDIA has no entry in
// MEDIA_BY_EXT, so it classified as a generic 'file' and rendered as a
// download-style link. Markdown is renderable content — it must route to the
// preview rail (which renders .md with a rendered/source toggle) instead.
describe('markdown documents delivered via MEDIA', () => {
  afterEach(cleanup)

  it('classifies markdown extensions as markdown documents', () => {
    expect(isMarkdownDocumentPath('/tmp/report.md')).toBe(true)
    expect(isMarkdownDocumentPath('/tmp/notes.markdown')).toBe(true)
    expect(isMarkdownDocumentPath('C:\\Users\\a\\report.MD')).toBe(true)
    expect(isMarkdownDocumentPath('/tmp/report.md?x=1')).toBe(true)
    expect(isMarkdownDocumentPath('/tmp/archive.zip')).toBe(false)
    expect(isMarkdownDocumentPath('/tmp/clip.mp4')).toBe(false)
    expect(isMarkdownDocumentPath('/tmp/README')).toBe(false)
  })

  it('renders a MEDIA .md as a preview attachment, not a download link', async () => {
    const href = mediaMarkdownHref('/home/user/out/report.md')

    render(<MarkdownTextContent isRunning={false} text={`[report.md](${href})`} />)

    // PreviewAttachment renders an "open preview" toggle button; the old
    // MediaAttachment 'file' fallback rendered a bare "Open ... " anchor.
    // Two buttons now: Download + Open preview (maintainer-requested).
    const buttons = await screen.findAllByRole('button')
    expect(buttons.length).toBe(2)
    expect(screen.getByText('Download')).toBeTruthy()
    expect(screen.queryByText(/^Loading /)).toBeNull()
    expect(screen.getByText('report.md')).toBeTruthy()
  })

  it('renders a non-markdown MEDIA file as a preview attachment too', async () => {
    // Extends #84951 to every non-media extension: PDFs, archives, data
    // files. MediaAttachment's kind==='file' branch was a degraded dead-end
    // (bare "Open ... " anchor, verified live with .pdf and .qzx7 — the
    // markdown-LINK path already gave these a proper file card). MEDIA:
    // must never render worse than a plain markdown link to the same file.
    const href = mediaMarkdownHref('/home/user/out/archive.zip')

    render(<MarkdownTextContent isRunning={false} text={`[archive.zip](${href})`} />)

    const buttons = await screen.findAllByRole('button')
    expect(buttons.length).toBe(2)
    expect(screen.getByText('Download')).toBeTruthy()
    expect(screen.getByText('archive.zip')).toBeTruthy()
    expect(screen.queryByText(/^Open archive/)).toBeNull()
  })

  it('renders a MEDIA pdf as a preview attachment', async () => {
    const href = mediaMarkdownHref('C:/Users/a/report.pdf')

    render(<MarkdownTextContent isRunning={false} text={`[report.pdf](${href})`} />)

    const buttons = await screen.findAllByRole('button')
    expect(buttons.length).toBe(2)
    expect(screen.getByText('report.pdf')).toBeTruthy()
  })

  it('still renders a genuine media-kind MEDIA ref through the enriched media card', async () => {
    // D4 (never-silent): file-kind refs route to the preview pipeline
    // (#97812), but genuine media kinds keep the MediaAttachment path —
    // now fed the href-carried byte size so reopened transcripts render
    // the card's size line without a registry hit. The card must never be
    // a silent dead anchor regardless of the file's current existence.
    const href = mediaMarkdownHref('/home/user/out/clip.mp4?~=1234')

    render(<MarkdownTextContent isRunning={false} text={`[clip.mp4](${href})`} />)

    expect(await screen.findByText('clip.mp4')).toBeTruthy()
    expect(screen.queryByText(/^Loading /)).toBeNull()
  })
})
