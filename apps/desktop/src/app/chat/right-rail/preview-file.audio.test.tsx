import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { PreviewTarget } from '@/store/preview'

import { LocalFilePreview } from './preview-file'

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

describe('LocalFilePreview audio playback', () => {
  beforeEach(() => {
    // A desktop bridge is what turns an audio path into a streamable source.
    window.hermesDesktop = {
      readFileText: vi.fn(async () => ({ binary: true, byteSize: 1_048_576, text: '' }))
    } as never
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('mounts an audio player instead of the binary refusal screen', async () => {
    const { container } = render(<LocalFilePreview reloadKey={0} target={audioTarget('/home/me/take.wav')} />)

    const player = await waitFor(() => {
      const element = container.querySelector('audio')

      expect(element).not.toBeNull()

      return element
    })

    expect(player?.getAttribute('controls')).not.toBeNull()
    // The refusal screen is the bug; it must not be what the user sees.
    expect(screen.queryByText(/binary file/i)).toBeNull()
  })

  it('streams the file through the range-capable media protocol', async () => {
    const { container } = render(<LocalFilePreview reloadKey={0} target={audioTarget('/home/me/take.wav')} />)

    const player = await waitFor(() => {
      const element = container.querySelector('audio')

      expect(element).not.toBeNull()

      return element
    })

    // A whole-file data URL would cap at the read limit and break seeking;
    // `hermes-media://stream/…` is the protocol that supports Range requests.
    expect(player?.getAttribute('src')).toBe(`hermes-media://stream/${encodeURIComponent('/home/me/take.wav')}`)
  })

  it('shows the file name so the user knows what is playing', async () => {
    render(<LocalFilePreview reloadKey={0} target={audioTarget('/home/me/take.wav')} />)

    await waitFor(() => {
      expect(screen.getByText('take.wav')).toBeTruthy()
    })
  })

  it('does not attempt a text read for an audio file', async () => {
    const readFileText = vi.fn(async () => ({ binary: true, byteSize: 1_048_576, text: '' }))
    window.hermesDesktop = { readFileText } as never

    render(<LocalFilePreview reloadKey={0} target={audioTarget('/home/me/take.wav')} />)

    await waitFor(() => {
      expect(screen.getByText('take.wav')).toBeTruthy()
    })

    // Sniffing a song as text is exactly what produced the garbage/refusal.
    expect(readFileText).not.toHaveBeenCalled()
  })

  it('shows a loading state, not the failure card, before the source resolves', () => {
    // Deliberately no waitFor: the source resolves after the first commit, and
    // that first commit is what used to flash "couldn't load" at the user.
    render(<LocalFilePreview reloadKey={0} target={audioTarget('/home/me/take.wav')} />)

    expect(screen.getByRole('status', { name: 'Loading preview' })).toBeTruthy()
    expect(screen.queryByText(/couldn't load this audio file/i)).toBeNull()
    expect(screen.queryByText('Open in external')).toBeNull()
  })

  it('shows the failure card only when playback actually errors', async () => {
    const { container } = render(<LocalFilePreview reloadKey={0} target={audioTarget('/home/me/take.wav')} />)

    const player = await waitFor(() => {
      const element = container.querySelector('audio')

      expect(element).not.toBeNull()

      return element!
    })

    fireEvent.error(player)

    expect(screen.getByText(/couldn't load this audio file/i)).toBeTruthy()
    expect(container.querySelector('audio')).toBeNull()
  })

  it('remounts the player when the preview watch bumps reloadKey', async () => {
    // An agent rewriting the .wav in place bumps reloadKey; the src string is
    // unchanged, so only a fresh <audio> element makes Chromium re-request it.
    const target = audioTarget('/home/me/take.wav')
    const { container, rerender } = render(<LocalFilePreview reloadKey={0} target={target} />)

    const first = await waitFor(() => {
      const element = container.querySelector('audio')

      expect(element).not.toBeNull()

      return element!
    })

    rerender(<LocalFilePreview reloadKey={1} target={target} />)

    await waitFor(() => {
      const element = container.querySelector('audio')

      expect(element).not.toBeNull()
      expect(element).not.toBe(first)
    })
  })

  it('retries after a failure when the file changes', async () => {
    const target = audioTarget('/home/me/take.wav')
    const { container, rerender } = render(<LocalFilePreview reloadKey={0} target={target} />)

    const player = await waitFor(() => {
      const element = container.querySelector('audio')

      expect(element).not.toBeNull()

      return element!
    })

    fireEvent.error(player)
    expect(screen.getByText(/couldn't load this audio file/i)).toBeTruthy()

    rerender(<LocalFilePreview reloadKey={1} target={target} />)

    await waitFor(() => {
      expect(container.querySelector('audio')).not.toBeNull()
    })
    expect(screen.queryByText(/couldn't load this audio file/i)).toBeNull()
  })
})
