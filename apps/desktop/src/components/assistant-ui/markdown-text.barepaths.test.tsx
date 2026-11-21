import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { IS_MAC } from '@/lib/keybinds/combo'
import { CHAT_LINK_ATTR, setChatLinksRequireModifier } from '@/store/chat-link-open'
import { $previewTabs, closeRightRail } from '@/store/preview'

import { MarkdownTextContent } from './markdown-text'

// #48313: a bare path or URL the agent hands over ("Report: /Users/me/r.md",
// `/tmp/out.md` in a code span) was dead text. It now opens like an authored
// link, inline so the sentence still reads as a sentence.
afterEach(() => {
  setChatLinksRequireModifier(false)
  closeRightRail()
  cleanup()
})

const PLAIN_CLICK = { detail: 1 }
const MODIFIER_CLICK = IS_MAC ? { detail: 1, metaKey: true } : { ctrlKey: true, detail: 1 }

describe('bare paths and URLs in assistant markdown', () => {
  it('renders a bare absolute path in prose as an inline link showing the path', () => {
    render(<MarkdownTextContent isRunning={false} text="Register is at /Users/me/dev/reports/register.md." />)

    const link = screen.getByRole('link', { name: '/Users/me/dev/reports/register.md' })

    expect(link.getAttribute('href')).toBe('/Users/me/dev/reports/register.md')
    expect(link.textContent).toBe('/Users/me/dev/reports/register.md')
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })

  it('renders a path that fills a code span as a link wrapping the code', () => {
    render(<MarkdownTextContent isRunning={false} text={'Open `/Users/me/dev/reports/register.md` when ready'} />)

    const link = screen.getByRole('link', { name: '/Users/me/dev/reports/register.md' })

    expect(link.querySelector('code')).not.toBeNull()
  })

  it('renders a URL that fills a code span as a link wrapping the code', () => {
    render(<MarkdownTextContent isRunning={false} text={'Server: `http://localhost:8931/styleguide.html`'} />)

    const link = screen.getByRole('link', { name: 'http://localhost:8931/styleguide.html' })

    expect(link.querySelector('code')).not.toBeNull()
    expect(link.getAttribute('href')).toBe('http://localhost:8931/styleguide.html')
  })

  it('leaves commands, relative paths, executables and fenced code as plain code', () => {
    render(
      <MarkdownTextContent
        isRunning={false}
        text={'Run `git checkout -- index.html`, edit `css/a.css`, not `/Applications/X.app`\n\n```sh\ncat /tmp/out.md\n```'}
      />
    )

    expect(screen.queryByRole('link')).toBeNull()
  })

  it('keeps an authored file link on the PreviewAttachment card', async () => {
    render(<MarkdownTextContent isRunning={false} text="Wrote [report](/home/user/report.md)" />)

    await screen.findByText('report.md')
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
    expect(screen.queryByRole('link')).toBeNull()
  })

  it('opens the preview pane when a bare path is clicked', async () => {
    render(<MarkdownTextContent isRunning={false} text="see /tmp/report.md" />)

    fireEvent.click(screen.getByRole('link', { name: '/tmp/report.md' }), PLAIN_CLICK)

    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))

    const target = $previewTabs.get()[0]!.target

    expect(target.kind === 'file' && target.path).toBe('/tmp/report.md')
  })

  it('does not linkify anything in foreign (preview-only) history', () => {
    render(
      <MarkdownTextContent
        isRunning={false}
        previewOnly
        text={'see /tmp/report.md and `/tmp/b.md` and `http://localhost:1/`'}
      />
    )

    expect(screen.queryByRole('link')).toBeNull()
  })
})

describe('Require ⌘/Ctrl-click to open chat links', () => {
  it('ignores a plain click and opens on the platform modifier, for paths and path chips', async () => {
    setChatLinksRequireModifier(true)
    render(<MarkdownTextContent isRunning={false} text={'see /tmp/a.md and `/tmp/b.md`'} />)

    const prose = screen.getByRole('link', { name: '/tmp/a.md' })
    const chip = screen.getByRole('link', { name: '/tmp/b.md' })

    fireEvent.click(prose, PLAIN_CLICK)
    fireEvent.click(chip, PLAIN_CLICK)
    await Promise.resolve()
    expect($previewTabs.get()).toHaveLength(0)

    fireEvent.click(chip, MODIFIER_CLICK)
    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
  })

  it('still opens on a keyboard activation', async () => {
    setChatLinksRequireModifier(true)
    render(<MarkdownTextContent isRunning={false} text="see /tmp/report.md" />)

    fireEvent.click(screen.getByRole('link', { name: '/tmp/report.md' }), { detail: 0 })

    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
  })

  it('gates authored URLs and URL chips, and opens them in-app on the modifier', async () => {
    setChatLinksRequireModifier(true)
    render(<MarkdownTextContent isRunning={false} text={'See [docs](https://example.com/guide) or `http://localhost:8931/`'} />)

    const authored = screen.getByRole('link', { name: 'docs' })
    const chip = screen.getByRole('link', { name: 'http://localhost:8931/' })

    expect(authored.hasAttribute(CHAT_LINK_ATTR)).toBe(true)
    expect(chip.hasAttribute(CHAT_LINK_ATTR)).toBe(true)

    fireEvent.click(authored, PLAIN_CLICK)
    fireEvent.click(chip, PLAIN_CLICK)
    await Promise.resolve()
    expect($previewTabs.get()).toHaveLength(0)

    fireEvent.click(authored, MODIFIER_CLICK)
    await waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe('https://example.com/guide'))
  })

  it('leaves the PreviewAttachment card on an ordinary click', async () => {
    setChatLinksRequireModifier(true)
    render(<MarkdownTextContent isRunning={false} text="Wrote [report](/home/user/report.md)" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Open preview' }))

    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
  })
})
