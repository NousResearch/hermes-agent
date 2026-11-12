import { describe, expect, it } from 'vitest'

import { isMarkdownDocumentPath, mediaKind, mediaMime } from './media'

describe('media kind for native paths that contain # or ?', () => {
  it('keeps the extension of a native path with a literal #', () => {
    const path = 'C:/Users/me/Music/track #07 - song.m4a'

    expect(mediaKind(path)).toBe('audio')
    expect(mediaMime(path)).toBe('audio/mp4')
    expect(mediaKind('C:\\Users\\me\\Music\\#1 hit.mp3')).toBe('audio')
    expect(mediaKind('/home/me/clips/take #2.mp4')).toBe('video')
  })

  it('still ignores the query and fragment of a URL', () => {
    expect(mediaKind('https://example.com/a.mp3?sig=1#t=10')).toBe('audio')
    expect(mediaKind('file:///C:/Music/a.wav#frag')).toBe('audio')
    expect(mediaKind('https://cdn.example/notes.txt?file=song.mp3')).toBe('file')
    expect(isMarkdownDocumentPath('/tmp/report.md?x=1')).toBe(true)
  })

  it('reads the extension from the file name, not from a dotted folder', () => {
    expect(mediaKind('C:/Users/me/v1.2/README')).toBe('file')
    expect(mediaKind('C:/Users/me/constructor')).toBe('file')
    expect(isMarkdownDocumentPath('C:/notes/#ideas.md')).toBe(true)
  })
})
