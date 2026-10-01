import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'
import { $connection } from '@/store/session'

import { MarkdownImage, MarkdownTextContent, MessageTextContent } from './markdown-text'

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

  it('keeps foreign-history images, local-file links and a real registered directive inert', async () => {
    const live = vi.fn(() => <div data-testid="foreign-live-card">Live action</div>)

    const dispose = registry.register({ id: 'test:foreign-history', area: TRANSCRIPT_DIRECTIVE_AREA, source: 'plugin:test',
      data: { name: 'foreign-history', render: live } satisfies TranscriptDirectiveContribution })

    try {
      // Positive control: this exact registered directive mounts in ordinary chat.
      const ordinary = render(<MessageTextContent media={false} text="::foreign-history" />)
      await screen.findByTestId('foreign-live-card')
      expect(live).toHaveBeenCalled()
      ordinary.unmount(); live.mockClear(); api.mockClear()

      const foreign = render(<MessageTextContent media={false} previewOnly text={
        `![Foreign image](${REMOTE_IMAGE_PATH})\n\n[Owner notes](/home/user/project/notes.md)\n\n::foreign-history\n\nMEDIA:/home/user/project/private.mp3`
      } />)

      await screen.findByText('Foreign image')
      expect(screen.getByText('Owner notes')).toBeTruthy()
      expect(foreign.container.textContent).toContain('::foreign-history')
      expect(foreign.container.textContent).toContain('MEDIA:/home/user/project/private.mp3')
      expect(foreign.container.querySelector('a, img, video, audio, button')).toBeNull()
      expect(screen.queryByTestId('foreign-live-card')).toBeNull()
      expect(live).not.toHaveBeenCalled()
      expect(api).not.toHaveBeenCalled()
    } finally {dispose()}
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

describe('MessageTextContent MEDIA directives', () => {
  afterEach(cleanup)

  it('renders a raw audio MEDIA directive through the canonical player instead of exposing the directive', async () => {
    const { container } = render(<MessageTextContent text="MEDIA:/tmp/group-voice.mp3" />)

    await waitFor(() => expect(container.querySelector('audio[controls]')).not.toBeNull())
    expect(container.textContent).not.toContain('MEDIA:')
  })
})
