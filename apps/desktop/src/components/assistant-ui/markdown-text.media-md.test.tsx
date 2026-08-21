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
    // MediaAttachment 'file' fallback rendered a bare "Open ..." anchor.
    expect(await screen.findByRole('button')).toBeTruthy()
    expect(screen.queryByText(/^Loading /)).toBeNull()
    expect(screen.getByText('report.md')).toBeTruthy()
  })

  it('still renders a non-markdown MEDIA file through the media fallback', async () => {
    const href = mediaMarkdownHref('/home/user/out/archive.zip')

    render(<MarkdownTextContent isRunning={false} text={`[archive.zip](${href})`} />)

    expect(await screen.findByText(/archive\.zip/)).toBeTruthy()
    expect(screen.queryByRole('button')).toBeNull()
  })
})
