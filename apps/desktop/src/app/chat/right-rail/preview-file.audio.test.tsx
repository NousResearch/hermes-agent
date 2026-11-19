import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setSessionOwnerResolver } from '@/api/client'
import type { PreviewTarget } from '@/store/preview'

import { AudioFilePreview } from './preview-file'

/**
 * Regression for #audio-preview: a `.wav` selected in the file browser showed
 * the "This looks like a binary file" refusal screen instead of playing.
 *
 * The rail already owns everything playback needs (`lib/media.ts` classifies
 * the path, `resolveMediaPlaybackSrc` yields a Range-capable
 * `hermes-media://stream/…` source), so these tests assert the two things the
 * bug broke: the refusal gate must not swallow audio, and the pane must mount a
 * real player for it.
 */

function audioTarget(path: string): PreviewTarget {
  return {
    binary: true,
    byteSize: 1_048_576,
    kind: 'file',
    label: path.split('/').pop() || path,
    path,
    previewKind: 'audio',
    source: path,
    url: `file://${path}`
  }
}

describe('AudioFilePreview playback', () => {
  afterEach(() => {
    cleanup()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('plays an audio file without reading it as text', async () => {
    const readFileText = vi.fn(async () => ({ binary: true, byteSize: 1_048_576, text: '' }))
    window.hermesDesktop = { readFileText } as never

    const { container } = render(<AudioFilePreview reloadKey={0} target={audioTarget('/home/me/take.wav')} />)

    const player = await waitFor(() => {
      const element = container.querySelector('audio')

      expect(element).not.toBeNull()

      return element!
    })

    // A Range-capable stream, never a capped data URL; and no text sniff (that produced the refusal).
    expect(player.getAttribute('src')).toBe(`hermes-media://stream/${encodeURIComponent('/home/me/take.wav')}`)
    expect(screen.queryByText(/binary file/i)).toBeNull()
    expect(readFileText).not.toHaveBeenCalled()
  })

  it("streams through the tab's own session, as the transcript player does", async () => {
    // A Docker agent's /workspace clip lives in its session's sandbox: without the session the
    // stream request names no session and the gateway looks for the file on the host.
    setSessionOwnerResolver(id => (id === 'sess-docker' ? { profile: 'coder' } : null))
    const getGatewayFileStreamUrl = vi.fn(async () => 'hermes-media://ticket/abc')
    window.hermesDesktop = { getGatewayFileStreamUrl } as never

    const { container } = render(
      <AudioFilePreview reloadKey={0} sessionId="sess-docker" target={audioTarget('/workspace/out/take.wav')} />
    )

    await waitFor(() => expect(container.querySelector('audio')?.getAttribute('src')).toBe('hermes-media://ticket/abc'))
    expect(getGatewayFileStreamUrl).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/workspace/out/take.wav', profile: 'coder', sessionId: 'sess-docker' })
    )
    setSessionOwnerResolver(null)
  })
})
