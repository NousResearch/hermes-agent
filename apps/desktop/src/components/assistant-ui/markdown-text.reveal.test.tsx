import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { renderMediaTags } from '@/lib/chat-messages'
import { $connection } from '@/store/session'

import { MarkdownTextContent } from './markdown-text'

// Regression for the dead "Open Link" on MEDIA file outputs: the file branch
// of MediaAttachment rendered with a literal `#` href, so the native
// right-click "Open Link"/"Copy Link" resolved to the app's own page URL
// instead of the file. The file branch now carries a real file:// href (so the
// native menu and "Copy Link" work again) and a "Reveal in Finder" action for
// local connections (the file lives on this disk, so the shell can reveal it).
describe('MediaAttachment file outputs', () => {
  const revealPath = vi.fn().mockResolvedValue(true)
  const openExternal = vi.fn()
  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api: vi.fn(), openExternal, revealPath }
    })
    $connection.set(null)
    revealPath.mockClear()
    openExternal.mockClear()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })
  })

  it('renders a real file href plus a reveal-in-finder action for MEDIA files', async () => {
    render(<MarkdownTextContent isRunning={false} text={renderMediaTags('MEDIA:/tmp/report.pdf')} />)

    const openLink = await screen.findByRole('link', { name: /Open report\.pdf/ })
    expect(openLink.getAttribute('href')).toBe('file:///tmp/report.pdf')

    const reveal = screen.getByRole('button', { name: 'Reveal in Finder' })
    fireEvent.click(reveal)
    await waitFor(() => expect(revealPath).toHaveBeenCalledWith('/tmp/report.pdf'))
  })

  it('keeps left-click opening the file (not navigating the anchor)', async () => {
    render(<MarkdownTextContent isRunning={false} text={renderMediaTags('MEDIA:/tmp/report.pdf')} />)

    const openLink = await screen.findByRole('link', { name: /Open report\.pdf/ })
    fireEvent.click(openLink)

    // useOpenMediaFile routes to openExternalLink(file://…) on a local
    // connection; the bridge receives the file URL.
    await waitFor(() =>
      expect(window.hermesDesktop?.openExternal).toHaveBeenCalledWith('file:///tmp/report.pdf')
    )
  })

  it('hides reveal-in-finder on a remote gateway', async () => {
    $connection.set({ mode: 'remote', profile: 'remote-work' } as never)
    render(<MarkdownTextContent isRunning={false} text={renderMediaTags('MEDIA:/tmp/report.pdf')} />)

    await screen.findByRole('link', { name: /Open report\.pdf/ })
    expect(screen.queryByRole('button', { name: 'Reveal in Finder' })).toBeNull()
  })
})
