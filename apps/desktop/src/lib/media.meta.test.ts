import { beforeEach, describe, expect, it } from 'vitest'

import { resetMediaDeliverables } from '@/lib/media-store'

import { mediaDisplayLabel, mediaHrefWithSize, mediaKindWithMeta, mediaPathAndSizeFromMarkdownHref } from './media'

describe('meta-aware media helpers (M4)', () => {
  beforeEach(() => {
    resetMediaDeliverables()
  })

  it('mediaKindWithMeta prefers the event kind over the extension table', () => {
    expect(mediaKindWithMeta('/tmp/tone.oga', { kind: 'audio', path: '/tmp/tone.oga', receivedAt: 1 })).toBe('audio')
    expect(mediaKindWithMeta('/tmp/tone.oga', null)).toBe('file')
  })

  it('mediaDisplayLabel carries the human size when the event reported one', () => {
    expect(mediaDisplayLabel('/tmp/clip.png', { kind: 'image', path: '/tmp/clip.png', receivedAt: 1, size: 1234 })).toBe(
      'Image · 1.2 KB: clip.png'
    )
    // No metadata → legacy label, byte-for-byte.
    expect(mediaDisplayLabel('/tmp/clip.png', null)).toBe('Image: clip.png')
  })

  it('formatMediaSize covers the unit ladder and refuses garbage', async () => {
    const { formatMediaSize } = await import('./media')

    expect(formatMediaSize(0)).toBe('0 B')
    expect(formatMediaSize(999)).toBe('999 B')
    expect(formatMediaSize(1234)).toBe('1.2 KB')
    expect(formatMediaSize(5_300_000)).toBe('5.3 MB')
    expect(formatMediaSize(undefined)).toBeNull()
    expect(formatMediaSize(-5)).toBeNull()
  })

  it('mediaHrefWithSize appends the ~= query only for a real size', () => {
    expect(mediaHrefWithSize('/tmp/clip.png', 1234)).toBe('#media:%2Ftmp%2Fclip.png?~=1234')
    expect(mediaHrefWithSize('/tmp/clip.png', undefined)).toBe('#media:%2Ftmp%2Fclip.png')
    expect(mediaHrefWithSize('/tmp/clip.png', -3)).toBe('#media:%2Ftmp%2Fclip.png')
  })

  it('mediaPathAndSizeFromMarkdownHref round-trips the size query', () => {
    expect(mediaPathAndSizeFromMarkdownHref('#media:%2Ftmp%2Fclip.png?~=1234')).toEqual({
      path: '/tmp/clip.png',
      size: 1234
    })
    expect(mediaPathAndSizeFromMarkdownHref('#media:%2Ftmp%2Fclip.png')).toEqual({ path: '/tmp/clip.png' })
    expect(mediaPathAndSizeFromMarkdownHref('#preview:%2Ftmp%2Fnotes.md')).toBeNull()
  })
})
