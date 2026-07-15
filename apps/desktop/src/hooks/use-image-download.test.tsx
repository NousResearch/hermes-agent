// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { imageFilename, useImageDownload } from './use-image-download'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: {
        downloadStarted: 'Download started',
        imageDownloadFailed: 'Image download failed',
        imageSaved: 'Image saved',
        restartToSaveImages: 'Restart to save images',
        restartToUseSaveImage: 'Restart to use Save Image'
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

describe('useImageDownload', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('passes the original media filename separately from a data URL preview', async () => {
    const saveImageFromUrl = vi.fn(async () => true)
    vi.stubGlobal('window', {
      hermesDesktop: { saveImageFromUrl },
      location: { href: 'http://localhost/' }
    })

    const src = 'data:image/png;base64,ZHVtbXk='
    const { result } = renderHook(() => useImageDownload(src, 'generated-image.png'))

    await act(async () => {
      await result.current.download()
    })

    expect(saveImageFromUrl).toHaveBeenCalledWith(src, 'generated-image.png')
  })

  it.each(['my image.png', 'generated#1.png', 'generated?1.png', '日本語.png'])(
    'treats suggested filename %s as a basename rather than a URL',
    suggestedFilename => {
      expect(imageFilename('data:image/png;base64,ZHVtbXk=', suggestedFilename)).toBe(suggestedFilename)
    }
  )
})
