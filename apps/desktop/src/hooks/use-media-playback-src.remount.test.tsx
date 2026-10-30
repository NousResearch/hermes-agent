import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as Media from '@/lib/media'
import { resolveMediaPlaybackSrc } from '@/lib/media'

import { useMediaPlaybackSrc } from './use-media-playback-src'

// Transcript rows remount routinely (a finished turn re-renders its markdown,
// a reconnect re-resumes the chat). A new stream ticket per mount made the
// browser fetch the clip again under a new URL, so a remount reuses the URL.

vi.mock('@/lib/media', async importOriginal => ({
  ...(await importOriginal<typeof Media>()),
  resolveMediaPlaybackSrc: vi.fn()
}))

const resolve = vi.mocked(resolveMediaPlaybackSrc)

function Player() {
  const { onError, src } = useMediaPlaybackSrc('/srv/clip.mp4')

  return src ? <video data-testid="video" onError={onError} preload="metadata" src={src} /> : null
}

const src = () => screen.getByTestId('video').getAttribute('src')

beforeEach(() => {
  let minted = 0
  resolve.mockImplementation(async () => `http://127.0.0.1/api/files/stream?ticket=t${++minted}`)
})

afterEach(() => {
  resolve.mockReset()
})

it('a remounted clip keeps its stream URL until that URL fails', async () => {
  const first = render(<Player />)
  await waitFor(() => expect(src()).toMatch(/ticket=t1$/))
  first.unmount()

  const second = render(<Player />)
  expect(src()).toMatch(/ticket=t1$/)
  expect(resolve).toHaveBeenCalledTimes(1)

  // A source that fails before it ever loaded is a real failure: the next mount resolves afresh.
  fireEvent.error(screen.getByTestId('video'))
  second.unmount()
  render(<Player />)
  await waitFor(() => expect(src()).toMatch(/ticket=t2$/))
})
