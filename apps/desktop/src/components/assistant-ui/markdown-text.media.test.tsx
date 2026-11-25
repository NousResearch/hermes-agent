import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

  it('keeps foreign-history images, file links and a real registered directive inert', async () => {
    const live = vi.fn(() => <div data-testid="foreign-live-card">Live action</div>)

    const dispose = registry.register({
      id: 'test:foreign-history',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      source: 'plugin:test',
      data: { name: 'foreign-history', render: live } satisfies TranscriptDirectiveContribution
    })

    try {
      // Positive control: this exact registered directive mounts in ordinary chat.
      const ordinary = render(<MessageTextContent media={false} text="::foreign-history" />)
      await screen.findByTestId('foreign-live-card')
      ordinary.unmount()
      live.mockClear()
      api.mockClear()

      const foreign = render(
        <MessageTextContent
          media={false}
          previewOnly
          text={`![Foreign image](${REMOTE_IMAGE_PATH})\n\n[Owner notes](/home/user/project/notes.md)\n\n::foreign-history\n\nMEDIA:/home/user/project/private.mp3\n\n[Relative notes](notes.md) [File URI](file:///home/peer/private.md) [Credential URL](https://user:secret@example.com/private) [Unsafe script](javascript:alert(1))`}
        />
      )

      await screen.findByText('Foreign image')
      expect(screen.getByText('Owner notes')).toBeTruthy()
      expect(foreign.container.textContent).toContain('::foreign-history')
      expect(foreign.container.textContent).toContain('MEDIA:/home/user/project/private.mp3')
      expect(foreign.container.querySelector('a, img, video, audio, button')).toBeNull()
      expect(screen.queryByTestId('foreign-live-card')).toBeNull()
      expect(live).not.toHaveBeenCalled()
      expect(api).not.toHaveBeenCalled()
    } finally {
      dispose()
    }
  })

  it('opens only an explicit web link from foreign history, with only the sanitized address and the safe target', async () => {
    const view = render(<MessageTextContent media={false} previewOnly text={'Read [the release notes](https://example.com/notes?x=1 "Release notes").'} />)
    const link = await screen.findByRole('link', { name: 'the release notes' })
    expect(link.getAttribute('href')).toBe('https://example.com/notes?x=1')
    expect(view.container.querySelectorAll('a')).toHaveLength(1)
    // Nothing the renderer hands the link component (its node, a title, a target) reaches the anchor.
    expect(link.getAttributeNames().sort()).toEqual(['class', 'href', 'rel', 'target'])
    expect([link.getAttribute('target'), link.getAttribute('rel')]).toEqual(['_blank', 'noopener noreferrer'])
    expect(api).not.toHaveBeenCalled()
  })

  // The parsed hostname folds decimal, octal, hex, dotted-unicode and percent-encoded spellings;
  // bracketed IPv6 hosts are already blocked by the renderer's URL hardening.
  it.each([
    'http://127.0.0.1:9119/api/status', 'http://2130706433/', 'http://0177.0.0.1/', 'http://0x7f000001/',
    'http://①②⑦.0.0.1/', 'https://%6c%6f%63%61%6c%68%6f%73%74/', 'http://localhost./', 'http://dash.localhost/',
    'http://0.0.0.0:8080/', 'http://[::1]/', 'http://[::ffff:127.0.0.1]/', 'http://[fe80::1]/', 'http://[fd12:3456::1]/',
    'http://169.254.169.254/latest/meta-data/', 'http://10.1.2.3/', 'http://172.16.0.1/', 'http://192.168.1.1/'
  ])('keeps a foreign link to a local or private host as text: %s', async href => {
    const view = render(<MessageTextContent media={false} previewOnly text={`Open [the dashboard](${href}) or [the docs](https://example.com/docs).`} />)
    await screen.findByRole('link', { name: 'the docs' })
    expect(screen.getByText(/the dashboard/).closest('a')).toBeNull()
    expect(view.container.querySelectorAll('a')).toHaveLength(1)
  })

  it('shows a foreign svg fence as its source instead of drawing it', async () => {
    const fence = '```svg\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 4"><circle cx="2" cy="2" r="1"/></svg>\n```'
    // Positive control: ordinary chat draws the same fence.
    const ordinary = render(<MessageTextContent text={fence} />)
    await waitFor(() => expect(ordinary.container.querySelector('circle')).toBeTruthy())
    ordinary.unmount()

    const foreign = render(<MessageTextContent media={false} previewOnly text={fence} />)
    await waitFor(() => expect(foreign.container.textContent).toContain('<circle'))
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(foreign.container.querySelector('svg circle')).toBeNull()
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

// Webapp audio/video play from stream-ticket URLs that expire. An element
// that already loaded needs a fresh ticket for its next range request (a seek,
// or play after a long pause) instead of degrading to "Open file" for good.
describe('MarkdownImage stream-ticket renewal', () => {
  const mint = vi.fn()
  let originalDesktop: typeof window.hermesDesktop

  const audioSrc = (container: HTMLElement) => container.querySelector('audio')?.getAttribute('src')

  beforeEach(() => {
    let minted = 0

    mint.mockReset().mockImplementation(async () => `https://hermes.test/api/files/stream?ticket=t${++minted}`)
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { getGatewayFileStreamUrl: mint } })
  })

  afterEach(() => {
    cleanup()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
  })

  it('re-mints the source after a loaded element errors and resumes where it stopped', async () => {
    const { container } = render(<MarkdownImage alt="note" src="/srv/media/note.mp3" />)

    await waitFor(() => expect(audioSrc(container)).toBe('https://hermes.test/api/files/stream?ticket=t1'))
    const audio = container.querySelector('audio')!
    fireEvent.loadedMetadata(audio)
    audio.currentTime = 42
    fireEvent.error(audio)

    await waitFor(() => expect(audioSrc(container)).toBe('https://hermes.test/api/files/stream?ticket=t2'))
    audio.currentTime = 0
    fireEvent.loadedMetadata(audio)
    expect(audio.currentTime).toBe(42)
    expect(screen.queryByText('Open audio file')).toBeNull()
  })

  it('offers the file instead when a source fails before it ever loads', async () => {
    const { container } = render(<MarkdownImage alt="note" src="/srv/media/note.mp3" />)

    await waitFor(() => expect(audioSrc(container)).toBe('https://hermes.test/api/files/stream?ticket=t1'))
    fireEvent.error(container.querySelector('audio')!)

    expect(await screen.findByText('Open audio file')).toBeTruthy()
    expect(mint).toHaveBeenCalledTimes(1)
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
