import { describe, expect, it } from 'vitest'

import { isAudioFilePath, mediaKind } from './media'

// The preview rail and the chat transcript must agree on what an audio file IS.
// `mediaKind()` already owns the extension→kind mapping for MEDIA: delivery;
// this predicate is that same table asked a yes/no question, so a format added
// to MEDIA_BY_EXT cannot be playable in chat but refused in the preview.
describe('isAudioFilePath', () => {
  it.each(['song.wav', 'song.mp3', 'song.flac', 'song.m4a', 'song.ogg', 'song.opus'])(
    'treats %s as audio',
    path => {
      expect(isAudioFilePath(path)).toBe(true)
    }
  )

  // Audio and video share the media pipeline but not the player affordances;
  // a video container must never be routed to the audio branch.
  it.each(['clip.mp4', 'clip.mov', 'clip.mkv', 'clip.avi', 'clip.webm'])(
    'does not treat the video file %s as audio',
    path => {
      expect(isAudioFilePath(path)).toBe(false)
    }
  )

  it('agrees with mediaKind for every mapped extension', () => {
    const samples = ['a.wav', 'a.mp3', 'a.flac', 'a.m4a', 'a.ogg', 'a.opus', 'a.mp4', 'a.mkv', 'a.png', 'a.pdf']

    for (const path of samples) {
      expect(isAudioFilePath(path)).toBe(mediaKind(path) === 'audio')
    }
  })

  it('handles uppercase extensions and query suffixes', () => {
    expect(isAudioFilePath('/tmp/TAKE.WAV')).toBe(true)
    expect(isAudioFilePath('https://example.com/track.wav?token=abc')).toBe(true)
  })

  it('rejects extension-less and non-media paths', () => {
    expect(isAudioFilePath('/tmp/notes.txt')).toBe(false)
    expect(isAudioFilePath('/tmp/Makefile')).toBe(false)
    expect(isAudioFilePath('')).toBe(false)
  })
})
