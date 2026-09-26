import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setAlwaysExternalLinks } from '@/store/external-links'
import { $previewTabs, closeRightRail } from '@/store/preview'
import { $connection } from '@/store/session'

import { MarkdownTextContent } from './markdown-text'

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
  afterEach(() => {
    cleanup()
    setAlwaysExternalLinks(false)
    closeRightRail()
    $connection.set(null)
    vi.restoreAllMocks()
  })

  it('keeps a remote PDF in the gateway-backed pane instead of opening a broken local file URL', async () => {
    const openPreviewInBrowser = vi.fn().mockResolvedValue(undefined)
    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: vi.fn(async () => ({
          kind: 'file',
          label: 'report.pdf',
          previewKind: 'pdf',
          source: '/srv/report.pdf',
          url: 'file:///srv/report.pdf'
        })),
        openPreviewInBrowser
      }
    })
    setAlwaysExternalLinks(true)

    render(<MarkdownTextContent isRunning={false} text="[report](/srv/report.pdf)" />)
    expect(screen.queryByRole('button', { name: 'Open in browser' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))

    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
    expect(openPreviewInBrowser).not.toHaveBeenCalled()
  })

  it('routes the transcript Preview action to the system browser when the preference is enabled', async () => {
    const openPreviewInBrowser = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: vi.fn(async () => ({
          kind: 'file',
          label: 'report.html',
          previewKind: 'html',
          source: '/tmp/report.html',
          url: 'file:///tmp/report.html'
        })),
        openPreviewInBrowser
      }
    })
    setAlwaysExternalLinks(true)

    render(<MarkdownTextContent isRunning={false} text="[report](/tmp/report.html)" />)
    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))

    await waitFor(() => expect(openPreviewInBrowser).toHaveBeenCalledWith('file:///tmp/report.html'))
    expect($previewTabs.get()).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
  })

  it('offers an explicit browser button even when in-app preview is the default', async () => {
    const openPreviewInBrowser = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: vi.fn(async () => ({
          kind: 'file',
          label: 'report.html',
          previewKind: 'html',
          source: '/tmp/report.html',
          url: 'file:///tmp/report.html'
        })),
        openPreviewInBrowser
      }
    })

    render(<MarkdownTextContent isRunning={false} text="[report](/tmp/report.html)" />)
    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))

    await waitFor(() => expect(openPreviewInBrowser).toHaveBeenCalledWith('file:///tmp/report.html'))
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('routes an absolute file path link through the preview attachment', async () => {
    render(<MarkdownTextContent isRunning={false} text="Wrote it: [report](/home/user/report.md)" />)

    // PreviewAttachment paints the filename + an Open preview button —
    // that's the view-time door, not a dead <a>.
    await screen.findByText('report.md')
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
    expect(globalThis.document.querySelector('a[href="/home/user/report.md"]')).toBeNull()
  })

  it('routes file:// and ~/ links the same way', async () => {
    render(
      <MarkdownTextContent isRunning={false} text={'See [notes](file:///srv/data/notes.txt) and [todo](~/todo.md)'} />
    )

    await screen.findByText('notes.txt')
    await screen.findByText('todo.md')
    expect(screen.getAllByRole('button', { name: 'Open preview' })).toHaveLength(2)
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
    expect(globalThis.document.querySelector('a[href="#section-2"]')).not.toBeNull()
  })
})
