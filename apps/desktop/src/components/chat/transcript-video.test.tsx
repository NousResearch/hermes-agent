// Ported from block/buzz#7336: a rate picked in one video persists and seeds
// every later player; garbage stored values fall back to 1x.

import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.videoPlaybackSpeed'

const loadStore = () => import('@/store/video-playback-speed')
const loadComponent = () => import('./transcript-video')

describe('video playback speed preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(cleanup)

  it('persists a user rate change and seeds the next player from it', async () => {
    const { TranscriptVideo } = await loadComponent()
    const { container } = render(<TranscriptVideo src="file:///tmp/clip.mp4" />)
    const video = container.querySelector('video')!

    expect(video.playbackRate).toBe(1)

    // The user picks 2x in the native controls' rate menu.
    video.playbackRate = 2
    fireEvent.rateChange(video)

    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('2')

    // A player in a fresh renderer starts at the persisted rate.
    vi.resetModules()
    const fresh = await loadComponent()
    const next = render(<fresh.TranscriptVideo src="file:///tmp/other.mp4" />)

    expect(next.container.querySelector('video')!.playbackRate).toBe(2)
  })

  it('falls back to 1x for out-of-range or malformed stored values and drops the default key', async () => {
    window.localStorage.setItem(STORAGE_KEY, '250')
    let store = await loadStore()

    expect(store.$videoPlaybackSpeed.get()).toBe(1)

    vi.resetModules()
    window.localStorage.setItem(STORAGE_KEY, 'fast')
    store = await loadStore()

    expect(store.$videoPlaybackSpeed.get()).toBe(1)

    // Returning to the default removes the record instead of storing "1".
    store.setVideoPlaybackSpeed(1.5)
    store.setVideoPlaybackSpeed(1)

    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })
})
