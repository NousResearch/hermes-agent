import { describe, expect, it } from 'vitest'

import { mediaName } from './media'

describe('mediaName', () => {
  it('keeps non-ASCII characters in Windows drive-letter paths', () => {
    expect(mediaName('D:/Users/licat/Desktop/迷妃湖路段_payload.json')).toBe(
      '迷妃湖路段_payload.json'
    )
  })

  it('keeps non-ASCII characters in POSIX absolute paths', () => {
    expect(mediaName('/Users/licat/Desktop/迷妃湖路段_payload.json')).toBe(
      '迷妃湖路段_payload.json'
    )
  })

  it('keeps non-ASCII characters in home-relative paths', () => {
    expect(mediaName('~/Desktop/迷妃湖路段_payload.json')).toBe('迷妃湖路段_payload.json')
  })

  it('decodes percent-encoded URL pathnames', () => {
    expect(
      mediaName('https://example.com/%E8%BF%B7%E5%A6%83%E6%B9%96%E8%B7%AF%E6%AE%B5_payload.json')
    ).toBe('迷妃湖路段_payload.json')
  })

  it('returns the basename for a plain URL', () => {
    expect(mediaName('https://example.com/foo/bar.txt')).toBe('bar.txt')
  })

  it('returns the basename for a relative path', () => {
    expect(mediaName('foo/bar.txt')).toBe('bar.txt')
  })
})
