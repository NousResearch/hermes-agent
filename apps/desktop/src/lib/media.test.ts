import { describe, expect, it } from 'vitest'

import { mediaName } from './media'

describe('mediaName', () => {
  it('returns the basename of a native Windows path', () => {
    expect(mediaName('C:\\Users\\foo\\image.png')).toBe('image.png')
    expect(mediaName('C:\\a\\b\\c.png')).toBe('c.png')
  })

  it('returns the basename of a native forward-slash Windows path', () => {
    // `C:/…` is a drive-letter path, not a `scheme://` URL — the split must
    // still yield the trailing filename and not mistake `C:` for a URL scheme.
    expect(mediaName('C:/Users/foo/image.png')).toBe('image.png')
    expect(mediaName('C:/a/b/c.png')).toBe('c.png')
  })

  it('returns the basename of a UNC path', () => {
    expect(mediaName('\\\\server\\share\\image.png')).toBe('image.png')
    expect(mediaName('\\\\server\\share\\a\\b\\c.png')).toBe('c.png')
  })

  it('keeps POSIX, file:// and http(s) paths working', () => {
    expect(mediaName('/home/u/pic.png')).toBe('pic.png')
    expect(mediaName('file:///tmp/image.png')).toBe('image.png')
    expect(mediaName('https://a.com/b/c.png')).toBe('c.png')
  })
})
