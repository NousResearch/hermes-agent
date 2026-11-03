import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, type MockInstance, vi } from 'vitest'

import type * as Media from '@/lib/media'
import { resolveMediaPlaybackSrc } from '@/lib/media'

import { useMediaPlaybackSrc } from './use-media-playback-src'

vi.mock('@/lib/media', async importOriginal => ({
  ...(await importOriginal<typeof Media>()),
  resolveMediaPlaybackSrc: vi.fn()
}))

const resolve = vi.mocked(resolveMediaPlaybackSrc)

function Player({ path }: { path: string }) {
  const { failed, onError, onLoadedMetadata, src } = useMediaPlaybackSrc(path)

  return (
    <>
      {src && (
        <video data-testid="video" onError={onError} onLoadedMetadata={onLoadedMetadata} preload="none" src={src} />
      )}
      {failed && <span>failed</span>}
    </>
  )
}

const video = () => screen.getByTestId('video') as HTMLVideoElement

function pressPlay(element: HTMLVideoElement) {
  Object.defineProperty(element, 'paused', { configurable: true, value: false })
}

let play: MockInstance<HTMLMediaElement['play']>

beforeEach(() => {
  let minted = 0
  resolve.mockImplementation(async () => `http://127.0.0.1/api/files/stream?ticket=t${++minted}`)
  play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
  resolve.mockReset()
})

it('a video whose ticket expired before its first play gets a fresh ticket and plays', async () => {
  render(<Player path="/srv/clip.mp4" />)
  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t1$/))

  pressPlay(video())
  fireEvent.error(video())

  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t2$/))
  expect(screen.queryByText('failed')).toBeNull()
  await waitFor(() => expect(play).toHaveBeenCalledTimes(1))
})

it('a fresh source that also fails before loading is a real failure', async () => {
  render(<Player path="/srv/clip.mp4" />)
  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t1$/))

  fireEvent.error(video())
  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t2$/))
  fireEvent.error(video())

  await screen.findByText('failed')
  expect(resolve).toHaveBeenCalledTimes(2)
  expect(video().getAttribute('src')).toMatch(/ticket=t2$/)
})

it('a paused video gets its fresh ticket without being started', async () => {
  render(<Player path="/srv/clip.mp4" />)
  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t1$/))

  fireEvent.error(video())

  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t2$/))
  expect(play).not.toHaveBeenCalled()
})

it('a source that loaded and later expired is re-resolved', async () => {
  render(<Player path="/srv/clip.mp4" />)
  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t1$/))
  fireEvent.loadedMetadata(video())
  fireEvent.error(video())
  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t2$/))

  fireEvent.loadedMetadata(video())
  fireEvent.error(video())

  await waitFor(() => expect(video().getAttribute('src')).toMatch(/ticket=t3$/))
  expect(screen.queryByText('failed')).toBeNull()
})

it('a blob source never retries', async () => {
  resolve.mockImplementation(async () => 'blob:http://127.0.0.1/clip')
  render(<Player path="/srv/clip.mp4" />)
  await waitFor(() => expect(video().getAttribute('src')).toBe('blob:http://127.0.0.1/clip'))

  fireEvent.error(video())

  await screen.findByText('failed')
  expect(resolve).toHaveBeenCalledTimes(1)
})
