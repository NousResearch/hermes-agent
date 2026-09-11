import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { IS_MAC } from '@/lib/keybinds/combo'
import { INLINE_LINK_GATED_ATTR, setRequireModifierToOpenInlineLinks } from '@/store/inline-link-open'
import { $previewTabs, closeRightRail } from '@/store/preview'

import { MarkdownTextContent } from './markdown-text'

// Bare paths and URLs — the agent's most common shape ("Report:
// /Users/me/report.md", or `` `/tmp/out.md` `` in a code span) — were dead
// text: copyable, never clickable. #89472 made AUTHORED links
// (`[report](/path)`) open; this makes the unwrapped ones open too, as inline
// links so the sentence stays a sentence.
describe('bare paths and URLs in assistant markdown', () => {
  afterEach(() => {
    setRequireModifierToOpenInlineLinks(false)
    closeRightRail()
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders a bare absolute path in prose as an inline link showing the path', () => {
    render(<MarkdownTextContent isRunning={false} text="Register is open at /Users/me/dev/reports/register.md." />)

    const link = screen.getByRole('link', { name: '/Users/me/dev/reports/register.md' })

    expect(link.getAttribute('href')).toBe('/Users/me/dev/reports/register.md')
    // Inline, not the block PreviewAttachment card.
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
    // The sentence's full stop stays outside the link.
    expect(link.textContent).toBe('/Users/me/dev/reports/register.md')
  })

  it('renders a path that fills a code span as a link wrapping the code', () => {
    render(<MarkdownTextContent isRunning={false} text={'Open `/Users/me/dev/reports/register.md` when ready'} />)

    const link = screen.getByRole('link', { name: '/Users/me/dev/reports/register.md' })

    expect(link.querySelector('code')).not.toBeNull()
    expect(link.getAttribute('href')).toBe('/Users/me/dev/reports/register.md')
  })

  it('renders a URL that fills a code span as a link wrapping the code', () => {
    render(<MarkdownTextContent isRunning={false} text={'Server: `http://localhost:8931/styleguide.html`'} />)

    const link = screen.getByRole('link', { name: 'http://localhost:8931/styleguide.html' })

    expect(link.querySelector('code')).not.toBeNull()
    expect(link.getAttribute('href')).toBe('http://localhost:8931/styleguide.html')
  })

  it('leaves a command in a code span as plain code', () => {
    render(<MarkdownTextContent isRunning={false} text={'Run `git checkout -- index.html` to revert'} />)

    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByText('git checkout -- index.html').tagName).toBe('CODE')
  })

  it('leaves a relative path in a code span as plain code', () => {
    render(<MarkdownTextContent isRunning={false} text={'Edit `css/styleguide.css`'} />)

    expect(screen.queryByRole('link')).toBeNull()
  })

  it('does not linkify paths inside fenced code', () => {
    render(<MarkdownTextContent isRunning={false} text={'```sh\ncat /tmp/out.md\n```'} />)

    expect(screen.queryByRole('link')).toBeNull()
  })

  it('opens the preview pane when a bare path link is clicked', async () => {
    render(<MarkdownTextContent isRunning={false} text="see /tmp/report.md" />)

    fireEvent.click(screen.getByRole('link', { name: '/tmp/report.md' }))

    await waitFor(() => expect($previewTabs.get().length).toBe(1))

    const target = $previewTabs.get()[0]!.target

    expect(target.kind).toBe('file')
    expect(target.kind === 'file' && target.path).toBe('/tmp/report.md')
  })

  it('keeps an authored file link on the PreviewAttachment card (no double wrap)', async () => {
    render(<MarkdownTextContent isRunning={false} text="Wrote [report](/home/user/report.md)" />)

    await screen.findByText('report.md')
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
    expect(screen.queryByRole('link')).toBeNull()
  })
})

describe('require-modifier inline link clicks', () => {
  afterEach(() => {
    setRequireModifierToOpenInlineLinks(false)
    closeRightRail()
    cleanup()
    vi.restoreAllMocks()
  })

  const mouse = { detail: 1 }
  const modifier = IS_MAC ? { detail: 1, metaKey: true } : { detail: 1, ctrlKey: true }

  it('does not open a bare path on a regular mouse click', async () => {
    setRequireModifierToOpenInlineLinks(true)
    render(<MarkdownTextContent isRunning={false} text="see /tmp/report.md" />)

    fireEvent.click(screen.getByRole('link', { name: '/tmp/report.md' }), mouse)

    await Promise.resolve()
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('opens a bare path on the platform modifier click', async () => {
    setRequireModifierToOpenInlineLinks(true)
    render(<MarkdownTextContent isRunning={false} text="see /tmp/report.md" />)

    fireEvent.click(screen.getByRole('link', { name: '/tmp/report.md' }), modifier)

    await waitFor(() => expect($previewTabs.get().length).toBe(1))
    expect($previewTabs.get()[0]!.target.kind).toBe('file')
  })

  it('opens a keyboard-generated click on a bare path', async () => {
    setRequireModifierToOpenInlineLinks(true)
    render(<MarkdownTextContent isRunning={false} text="see /tmp/report.md" />)

    fireEvent.click(screen.getByRole('link', { name: '/tmp/report.md' }), { detail: 0 })

    await waitFor(() => expect($previewTabs.get().length).toBe(1))
  })

  it('treats a path code chip the same as a prose path', async () => {
    setRequireModifierToOpenInlineLinks(true)
    render(<MarkdownTextContent isRunning={false} text={'Open `/tmp/report.md` when ready'} />)

    const link = screen.getByRole('link', { name: '/tmp/report.md' })

    fireEvent.click(link, mouse)
    await Promise.resolve()
    expect($previewTabs.get()).toHaveLength(0)

    fireEvent.click(link, modifier)
    await waitFor(() => expect($previewTabs.get().length).toBe(1))
  })

  it('treats a URL code chip the same as a markdown URL', async () => {
    setRequireModifierToOpenInlineLinks(true)
    render(<MarkdownTextContent isRunning={false} text={'Server: `http://localhost:8931/styleguide.html`'} />)

    const link = screen.getByRole('link', { name: 'http://localhost:8931/styleguide.html' })

    fireEvent.click(link, mouse)
    await Promise.resolve()
    expect($previewTabs.get()).toHaveLength(0)

    fireEvent.click(link, modifier)
    await waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe('http://localhost:8931/styleguide.html'))
  })

  it('treats an authored markdown URL the same way', async () => {
    setRequireModifierToOpenInlineLinks(true)
    render(<MarkdownTextContent isRunning={false} text={'See [docs](https://example.com/guide)'} />)

    const link = screen.getByRole('link', { name: 'docs' })

    fireEvent.click(link, mouse)
    await Promise.resolve()
    expect($previewTabs.get()).toHaveLength(0)

    fireEvent.click(link, modifier)
    await waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe('https://example.com/guide'))
  })

  it('leaves PreviewAttachment card clicks on ordinary click', async () => {
    setRequireModifierToOpenInlineLinks(true)
    render(<MarkdownTextContent isRunning={false} text="Wrote [report](/home/user/report.md)" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Open preview' }))

    await waitFor(() => expect($previewTabs.get().length).toBe(1))
    expect($previewTabs.get()[0]!.target.kind).toBe('file')
  })

  it('marks prose paths, URL chips, path chips and authored URLs as cursor-gated', () => {
    setRequireModifierToOpenInlineLinks(true)

    const { rerender } = render(<MarkdownTextContent isRunning={false} text="see /tmp/report.md" />)

    expect(screen.getByRole('link', { name: '/tmp/report.md' }).hasAttribute(INLINE_LINK_GATED_ATTR)).toBe(true)

    rerender(<MarkdownTextContent isRunning={false} text={'Open `/tmp/report.md` when ready'} />)
    expect(screen.getByRole('link', { name: '/tmp/report.md' }).hasAttribute(INLINE_LINK_GATED_ATTR)).toBe(true)

    rerender(
      <MarkdownTextContent isRunning={false} text={'Server: `http://localhost:8931/styleguide.html`'} />
    )
    expect(
      screen.getByRole('link', { name: 'http://localhost:8931/styleguide.html' }).hasAttribute(INLINE_LINK_GATED_ATTR)
    ).toBe(true)

    rerender(<MarkdownTextContent isRunning={false} text={'See [docs](https://example.com/guide)'} />)
    expect(screen.getByRole('link', { name: 'docs' }).hasAttribute(INLINE_LINK_GATED_ATTR)).toBe(true)
  })
})
