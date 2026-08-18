import { describe, expect, it } from 'vitest'

import { downloadFilename, imageFilename } from './use-image-download'

describe('imageFilename', () => {
  it('takes the last path segment of a URL', () => {
    expect(imageFilename('https://v3.fal.media/files/kangaroo/pic.png')).toBe('pic.png')
  })

  it('falls back to "image" when there is no usable segment', () => {
    expect(imageFilename('https://example.com/')).toBe('image')
    expect(imageFilename(undefined)).toBe('image')
  })

  it('preserves the original filename when the display source is a data URL', () => {
    expect(imageFilename('data:image/png;base64,AAAA', 'slashstack-worktree-feedback-2026-08-18.png')).toBe(
      'slashstack-worktree-feedback-2026-08-18.png'
    )
  })
})

describe('downloadFilename', () => {
  it('keeps a name that already has a known image extension', () => {
    expect(downloadFilename('https://example.com/a/photo.jpg', 'image/jpeg')).toBe('photo.jpg')
    expect(downloadFilename('https://example.com/a/photo.webp', '')).toBe('photo.webp')
  })

  it('appends a MIME-derived extension to extensionless content hashes', () => {
    expect(downloadFilename('https://v3.fal.media/files/x/MKZV6h-RrKLVCOKp9bGfE_YuJPemAQ', 'image/jpeg')).toBe(
      'MKZV6h-RrKLVCOKp9bGfE_YuJPemAQ.jpg'
    )
    expect(downloadFilename('https://cdn.example.com/abc123', 'image/webp')).toBe('abc123.webp')
  })

  it('handles MIME parameters and unknown types', () => {
    expect(downloadFilename('https://cdn.example.com/abc123', 'image/png; charset=binary')).toBe('abc123.png')
    expect(downloadFilename('https://cdn.example.com/abc123', 'application/octet-stream')).toBe('abc123.png')
    expect(downloadFilename('https://cdn.example.com/abc123', undefined)).toBe('abc123.png')
  })

  it('uses a preferred semantic filename when the source has no filename', () => {
    expect(downloadFilename('data:image/png;base64,AAAA', 'image/png', 'slashstack-worktree-feedback-2026-08-18.png')).toBe(
      'slashstack-worktree-feedback-2026-08-18.png'
    )
  })

  it('does not treat a dotted hash suffix as an extension', () => {
    // A name like "photo.v2" has an extname but not a known image one — the
    // MIME extension still gets appended so the OS can open the file.
    expect(downloadFilename('https://cdn.example.com/photo.v2', 'image/png')).toBe('photo.v2.png')
  })
})
