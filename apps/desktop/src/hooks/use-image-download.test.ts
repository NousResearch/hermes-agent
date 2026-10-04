import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { mediaName } from '@/lib/media'
import { previewName } from '@/lib/preview-targets'

import { downloadFilename, imageFilename, useImageDownload } from './use-image-download'

describe('imageFilename', () => {
  it('uses the shared URL basename in browser downloads while preserving the fetch source', async () => {
    const cases = [
      ['https://example.com/my%20report.png?download=1#view', 'my report.png'],
      ['https://example.com/测试.png', '测试.png'],
      ['https://example.com/%E6%B5%8B%E8%AF%95.png', '测试.png'],
      ['https://example.com/a+b.png', 'a+b.png'],
      ['https://example.com/a%2520b.png', 'a%20b.png'],
      ['https://example.com/a%20b%2Fc.png', 'a b%2Fc.png'],
      ['https://example.com/a%20b%5Cc.png', 'a b%5Cc.png'],
      ['https://example.com/a%2fb%5cc.png', 'a%2fb%5cc.png'],
      ['https://example.com/a%252Fb%255Cc.png', 'a%2Fb%5Cc.png'],
      ['https://example.com/bad%ZZ.png', 'bad%ZZ.png'],
      ['https://example.com/100%.png', '100%.png'],
      ['https://example.com/bad%2.png', 'bad%2.png'],
      ['https://example.com/bad%E6%96.png', 'bad%E6%96.png'],
      ['https://example.com/bad%FF.png', 'bad%FF.png'],
      ['file:///tmp/a%2520b.png', 'a%20b.png'],
      ['file:///tmp/a%20b%2Fc.png', 'a b%2Fc.png'],
      ['file:///tmp/a%20b%5Cc.png', 'a b%5Cc.png'],
      ['file:///tmp/测试.png', '测试.png'],
      ['C:\\example\\测试%20.png', '测试%20.png'],
      ['C:/example/测试%20.png', '测试%20.png'],
      ['C:/example/a#b?.png', 'a#b?.png'],
      ['C:\\example\\a%2Fb%5Cc.png', 'a%2Fb%5Cc.png'],
      ['\\\\server\\share\\测试.png', '测试.png']
    ]

    for (const [src, name] of cases) {
      expect.soft(imageFilename(src), src).toBe(name)
      expect.soft(downloadFilename(src, 'image/png'), src).toBe(name)
      expect.soft(imageFilename(src), src).toBe(mediaName(src))
      expect.soft(imageFilename(src), src).toBe(previewName(src))
    }

    // Browser-relative sources still resolve against the document URL.
    expect.soft(imageFilename('/images/my%20photo.png?size=2#view')).toBe('my photo.png')
    expect.soft(imageFilename('images/my%20photo.png?size=2#view')).toBe('my photo.png')
    expect.soft(imageFilename('../')).toBe('image')
    expect.soft(imageFilename('////')).toBe('image')
    expect.soft(downloadFilename('https://example.com/测试%20image', 'image/webp')).toBe('测试 image.webp')

    const previousDesktop = window.hermesDesktop
    const fetchImage = vi.fn(async () => ({ ok: true, blob: async () => new Blob(['image'], { type: 'image/png' }) }))
    const filenames: string[] = []

    vi.useFakeTimers()
    vi.stubGlobal('fetch', fetchImage)
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:test-image')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      filenames.push(this.download)
    })
    window.hermesDesktop = {} as typeof window.hermesDesktop

    try {
      for (const [src, name] of cases.slice(0, 9)) {
        const { result, unmount } = renderHook(() => useImageDownload(src))

        try {
          await act(async () => result.current.download())
          expect(fetchImage).toHaveBeenLastCalledWith(src)
          expect(filenames.at(-1)).toBe(name)
        } finally {
          unmount()
        }
      }
    } finally {
      window.hermesDesktop = previousDesktop
      vi.runOnlyPendingTimers()
      vi.useRealTimers()
      vi.restoreAllMocks()
      vi.unstubAllGlobals()
    }
  })

  it('takes the last path segment of a URL', () => {
    expect(imageFilename('https://v3.fal.media/files/kangaroo/pic.png')).toBe('pic.png')
  })

  it('falls back to "image" when there is no usable segment', () => {
    expect(imageFilename('https://example.com/')).toBe('image')
    expect(imageFilename(undefined)).toBe('image')
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

  it('does not treat a dotted hash suffix as an extension', () => {
    // A name like "photo.v2" has an extname but not a known image one — the
    // MIME extension still gets appended so the OS can open the file.
    expect(downloadFilename('https://cdn.example.com/photo.v2', 'image/png')).toBe('photo.v2.png')
  })
})
