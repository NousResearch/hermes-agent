import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { renderMediaTags } from '@/lib/chat-messages'
import { $previewTarget, closeRightRail } from '@/store/preview'
import { $connection, $currentCwd } from '@/store/session'

import { MarkdownImage, MarkdownTextContent } from './markdown-text'

const REMOTE_IMAGE_PATH = '/home/user/project/images/remote-preview.png'
const REMOTE_IMAGE_DATA_URL = 'data:image/png;base64,cmVtb3RlLWltYWdl'

describe('MarkdownTextContent remote images', () => {
  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path.startsWith('/api/fs/read-data-url?')) {
      return { dataUrl: REMOTE_IMAGE_DATA_URL }
    }

    throw new Error(`unexpected path ${path}`)
  })

  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    api.mockClear()
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    $connection.set({ mode: 'remote', profile: 'remote-work' } as never)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })
  })

  it('passes the gateway bridge data URL through Streamdown to the zoomable image', async () => {
    render(<MarkdownTextContent isRunning={false} text={`![Remote preview](${REMOTE_IMAGE_PATH})`} />)

    const image = await screen.findByRole('img', { name: 'Remote preview' })

    expect(image.getAttribute('src')).toBe(REMOTE_IMAGE_DATA_URL)
    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-data-url?path=%2Fhome%2Fuser%2Fproject%2Fimages%2Fremote-preview.png',
      profile: 'remote-work'
    })
  })
})

// Regression for #40896: generated media often arrives as image markdown
// (`![clip](clip.mp4)`). A raw <img> with a video/audio source paints a
// broken-image icon even though the file is valid, so MarkdownImage must route
// video/audio sources to the proper <video>/<audio> element.
describe('MarkdownImage media routing', () => {
  afterEach(cleanup)

  it('renders a <video> (not a broken <img>) for a video source', async () => {
    const { container } = render(<MarkdownImage alt="clip" src="file:///tmp/clip.mp4" />)

    await waitFor(() => expect(container.querySelector('video')).not.toBeNull())
    expect(container.querySelector('img')).toBeNull()
  })

  it('renders an <audio> element for an audio source', async () => {
    const { container } = render(<MarkdownImage alt="note" src="file:///tmp/note.mp3" />)

    await waitFor(() => expect(container.querySelector('audio')).not.toBeNull())
    expect(container.querySelector('img')).toBeNull()
  })

  it('still renders an <img> for an image source', () => {
    const { container } = render(<MarkdownImage alt="pic" src="file:///tmp/pic.png" />)

    expect(container.querySelector('video')).toBeNull()
    expect(container.querySelector('audio')).toBeNull()
  })
})

describe('MarkdownTextContent local files', () => {
  const normalizePreviewTarget = vi.fn(async (target: string) => ({
    kind: 'file' as const,
    label: target.split(/[\\/]/).pop() || target,
    language: 'markdown',
    path: target,
    previewKind: 'text' as const,
    source: target,
    url: `file://${target.split('/').map(encodeURIComponent).join('/')}`
  }))

  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    normalizePreviewTarget.mockClear()
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { normalizePreviewTarget }
    })
    $connection.set({ mode: 'local', profile: 'default' } as never)
    $currentCwd.set('/Users/test/Documents/Hermes Projects/Monark-Inc/Bank-Charter')
    closeRightRail()
  })

  afterEach(() => {
    cleanup()
    closeRightRail()
    $connection.set(null)
    $currentCwd.set('')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })
  })

  it('opens the complete unquoted Markdown MEDIA path in PreviewAttachment when it contains spaces', async () => {
    const path =
      '/Users/test/Documents/Hermes Projects/Monark-Inc/Bank-Charter/.hermes/plans/2026-09-03_204030-monark-i13-plus-time-thesis-deep-research.md'

    render(<MarkdownTextContent isRunning={false} text={renderMediaTags(`MEDIA:${path}`)} />)

    expect(await screen.findByText('2026-09-03_204030-monark-i13-plus-time-thesis-deep-research.md')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    const openPreview = screen.getByRole('button', { name: 'Open preview' })

    expect(normalizePreviewTarget).not.toHaveBeenCalled()
    fireEvent.click(openPreview)
    await waitFor(() => expect($previewTarget.get()?.path).toBe(path))
    expect(normalizePreviewTarget).toHaveBeenCalledOnce()
    expect(normalizePreviewTarget).toHaveBeenCalledWith(
      path,
      '/Users/test/Documents/Hermes Projects/Monark-Inc/Bank-Charter'
    )
  })
})
