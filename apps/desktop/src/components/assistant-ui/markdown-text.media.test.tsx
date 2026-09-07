import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { $connection } from '@/store/session'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { MarkdownImage, MarkdownTextContent } from './markdown-text'

const REMOTE_IMAGE_PATH = '/home/user/project/images/remote-preview.png'
const REMOTE_IMAGE_DATA_URL = 'data:image/png;base64,cmVtb3RlLWltYWdl'

function sessionView(storedId: string): SessionView {
  return {
    kind: 'primary',
    $awaitingResponse: atom(false),
    $busy: atom(false),
    $cwd: atom('/srv/work'),
    $fast: atom(false),
    $lastVisibleIsUser: atom(false),
    $messages: atom([]),
    $messagesEmpty: atom(true),
    $model: atom(''),
    $ownerRoute: atom({ connectionId: 'remote-gateway', mode: 'remote', profile: 'assistant' }),
    $provider: atom(''),
    $reasoningEffort: atom(''),
    $runtimeId: atom(null),
    $storedId: atom(storedId),
    $turnStartedAt: atom(null)
  }
}

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

  it('routes raw Markdown images through the transcript owner instead of the foreground', async () => {
    $connection.set({ connectionId: 'local-device', mode: 'local', profile: 'default' } as never)

    render(
      <SessionViewProvider value={sessionView('remote-session')}>
        <MarkdownTextContent isRunning={false} text="![Remote preview](/srv/media/frame.png)" />
      </SessionViewProvider>
    )

    await screen.findByRole('img', { name: 'Remote preview' })
    expect(api).toHaveBeenCalledWith({
      connectionId: 'remote-gateway',
      path: '/api/fs/read-data-url?path=%2Fsrv%2Fmedia%2Fframe.png&session_id=remote-session',
      profile: 'assistant'
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

describe('MarkdownTextContent remote video actions', () => {
  const saveGatewayFile = vi.fn(async () => ({ saved: true }))

  beforeEach(() => {
    saveGatewayFile.mockClear()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { saveGatewayFile }
    })
    $connection.set({ connectionId: 'local-device', mode: 'local', profile: 'default' } as never)
    setSessionOwnerHint('remote-session', {
      connectionId: 'remote-gateway',
      mode: 'remote',
      profile: 'assistant'
    })
  })

  afterEach(() => {
    cleanup()
    clearAllSessionStates()
    _resetSessionOwnerHintsForTests()
    $connection.set(null)
  })

  it('offers an explicit download routed through the transcript owner', async () => {
    render(
      <SessionViewProvider value={sessionView('remote-session')}>
        <MarkdownTextContent
          isRunning={false}
          text="[Video: movie.mp4](#media:%2Fsrv%2Fmedia%2Fmovie.mp4)"
        />
      </SessionViewProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Download' }))

    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'remote-gateway',
        path: '/srv/media/movie.mp4',
        profile: 'assistant',
        sessionId: 'remote-session',
        suggestedName: 'movie.mp4'
      })
    )
  })

  it('routes colliding stored ids by the runtime bound to this transcript', async () => {
    setSessionOwnerHint('remote-session', {
      connectionId: 'other-gateway',
      mode: 'remote',
      profile: 'assistant'
    })
    render(
      <SessionViewProvider value={sessionView('remote-session')}>
        <MarkdownTextContent
          isRunning={false}
          text="[Video: movie.mp4](#media:%2Fsrv%2Fmedia%2Fmovie.mp4)"
        />
      </SessionViewProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Download' }))

    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'remote-gateway',
        path: '/srv/media/movie.mp4',
        profile: 'assistant',
        sessionId: 'remote-session',
        suggestedName: 'movie.mp4'
      })
    )
  })
})
