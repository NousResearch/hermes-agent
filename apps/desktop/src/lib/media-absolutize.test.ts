import { describe, expect, it } from 'vitest'

import { absolutizeMediaPath } from '@/lib/media'

describe('absolutizeMediaPath', () => {
  it('joins a relative path onto the session cwd', () => {
    expect(absolutizeMediaPath('final.mp4', '/home/u/work')).toBe('/home/u/work/final.mp4')
    expect(absolutizeMediaPath('./sub/a.mp4', '/home/u/work/')).toBe('/home/u/work/./sub/a.mp4')
  })

  it('uses backslash join for pure-Windows cwds', () => {
    expect(absolutizeMediaPath('final.mp4', 'C:\\Users\\u\\workdir')).toBe('C:\\Users\\u\\workdir\\final.mp4')
  })

  it('never touches absolute, home, drive or scheme paths', () => {
    for (const p of [
      '/a/b.mp4',
      '~/v/b.mp4',
      'C:\\a\\b.mp4',
      'C:/a/b.mp4',
      'https://x/y.mp4',
      'data:video/mp4;base64,AAA',
      'hermes-media://stream/x'
    ]) {
      expect(absolutizeMediaPath(p, '/home/u')).toBe(p)
    }
  })

  it('returns the path unchanged without a cwd', () => {
    expect(absolutizeMediaPath('final.mp4', null)).toBe('final.mp4')
    expect(absolutizeMediaPath('final.mp4', '')).toBe('final.mp4')
  })
})
