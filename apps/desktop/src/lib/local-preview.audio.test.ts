import { describe, expect, it } from 'vitest'

import { localPreviewTarget } from './local-preview'

// The renderer's classifier is the FALLBACK path (used when the Electron
// bridge is unavailable or older). It must still see an audio file for what it
// is: `.wav` classified as `text` is what produced the "This looks like a
// binary file" screen for a perfectly playable file.
describe('localPreviewTarget audio classification', () => {
  it('classifies a local .wav as audio, not text', () => {
    expect(localPreviewTarget('/home/me/take.wav')?.previewKind).toBe('audio')
  })

  it('classifies an audio file relative to the cwd', () => {
    expect(localPreviewTarget('samples/take.mp3', '/home/me')?.previewKind).toBe('audio')
  })

  it('keeps audio classified through a file:// URL', () => {
    expect(localPreviewTarget('file:///home/me/take.flac')?.previewKind).toBe('audio')
  })

  // Regression guard for the sibling branch: image/video classification is
  // decided by extension and must not be swallowed by the new audio check.
  it('still classifies images and unknown files as before', () => {
    expect(localPreviewTarget('/home/me/shot.png')?.previewKind).toBe('image')
    expect(localPreviewTarget('/home/me/notes.txt')?.previewKind).toBe('text')
  })

  // A video file is media, but the rail has no video kind yet — it must fall
  // through to the existing text/binary path rather than pretend to be audio.
  it('does not classify a video file as audio', () => {
    expect(localPreviewTarget('/home/me/clip.mp4')?.previewKind).not.toBe('audio')
  })
})
