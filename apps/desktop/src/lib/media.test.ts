import { describe, it, expect } from 'vitest'
import { mediaKind, mediaMime, mediaName } from '@/lib/media'

describe('mediaKind', () => {
  it('classifies local files by extension', () => {
    expect(mediaKind('/tmp/shot.png')).toBe('image')
    expect(mediaKind('/tmp/clip.jpg')).toBe('image')
    expect(mediaKind('/tmp/voice.mp3')).toBe('audio')
    expect(mediaKind('/tmp/movie.mp4')).toBe('video')
    expect(mediaKind('/tmp/notes.txt')).toBe('file')
  })

  it('treats extensionless remote http(s) URLs as images', () => {
    // Unsplash and similar CDNs serve images without a file extension
    // (e.g. images.unsplash.com/photo-…). Previously these fell through to
    // 'file' and MediaAttachment blocked them as "Image blocked: No description".
    expect(mediaKind('https://images.unsplash.com/photo-1506744038136-46273834b3fb')).toBe('image')
    expect(mediaKind('https://picsum.photos/200')).toBe('image')
    expect(mediaKind('http://example.com/img?id=123')).toBe('image')
  })

  it('still classifies unknown local paths as files', () => {
    expect(mediaKind('/tmp/no-extension')).toBe('file')
  })

  it('keeps remote URLs with a file extension as files, not images', () => {
    // A remote URL that ends in a recognized file extension (.pdf, .txt, …)
    // must NOT be coerced to an image just because it is remote. That would
    // bypass MediaAttachment's `file` path and attempt (and fail) thumbnail
    // loading on a document. Only genuinely extensionless pathnames are
    // treated as images.
    expect(mediaKind('https://example.test/report.pdf')).toBe('file')
    expect(mediaKind('https://example.test/notes.txt')).toBe('file')
    expect(mediaKind('https://images.unsplash.com/photo-123/report.pdf')).toBe('file')
  })
})

describe('mediaMime', () => {
  it('returns image mime for extensionless remote URLs', () => {
    expect(mediaMime('https://images.unsplash.com/photo-1506744038136-46273834b3fb')).toBe('image/')
  })

  it('returns octet-stream for remote URLs that have a file extension', () => {
    expect(mediaMime('https://example.test/report.pdf')).toBe('application/octet-stream')
  })
})

describe('mediaName', () => {
  it('uses the URL path segment for remote URLs', () => {
    expect(mediaName('https://images.unsplash.com/photo-1506744038136-46273834b3fb')).toBe(
      'photo-1506744038136-46273834b3fb'
    )
  })
})
